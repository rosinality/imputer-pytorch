import argparse
import io
import re
import os
import string
import wave
import pickle
from multiprocessing import Pool
from functools import partial

import lmdb
import librosa
import torch
from tqdm import tqdm

from audio import FilterbankFeature
from text import collapse_whitespace

re_pronunciation = re.compile(r'\((.*?)\)\/\((.*?)\)')
re_noise = re.compile(r'b\/|l\/|o\/|n\/')
table_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

PCM_CHANNELS = 1
PCM_BIT_DEPTH = 16
PCM_SAMPLING_RATE = 16000

N_META_CHAR = 3


def use_pronunciation(text):
    return re_pronunciation.sub(r'\2', text)


def remove_noise(text):
    return re_noise.sub(' ', text)


def remove_punctuation(text):
    return text.translate(table_punctuation)


def process_text(text):
    return collapse_whitespace(
        remove_punctuation(remove_noise(use_pronunciation(text)))
    ).strip()


def load_pcm(filename):
    with open(filename, 'rb') as f:
        pcmdata = f.read()

    wav_write = io.BytesIO()
    wav = wave.open(wav_write, 'wb')
    wav.setparams(
        (PCM_CHANNELS, PCM_BIT_DEPTH // 8, PCM_SAMPLING_RATE, 0, 'NONE', 'NONE')
    )
    wav.writeframes(pcmdata)

    wav_write.seek(0)
    wav, _ = librosa.load(wav_write, sr=PCM_SAMPLING_RATE)

    return wav


def load_text(filename):
    with open(filename, encoding='cp949') as f:
        return f.read()


def process_worker(filename, root):
    file = os.path.join(root, filename)
    wav = load_pcm(file + '.pcm')
    text = load_text(file + '.txt')

    wav_feat = wav_feature(torch.from_numpy(wav).unsqueeze(0), PCM_SAMPLING_RATE)
    text_feat = process_text(text)

    record = (wav_feat, text_feat, filename)

    return record


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('output', metavar='OUTPUT')

    args = parser.parse_args()

    speech_files = []

    wav_feature = FilterbankFeature(args.n_mels)

    for dirpath, dirs, files in os.walk(args.path):
        if len(dirs) == 0:
            speech_keys = set()

            for file in files:
                speech_keys.add(os.path.splitext(file)[0])

            speech_keys = list(sorted(speech_keys))

            relpath = os.path.relpath(dirpath, args.path)

            for key in speech_keys:
                speech_files.append(os.path.join(relpath, key))

    vocab = {}

    worker = partial(process_worker, root=args.path)

    with Pool(processes=8) as pool, lmdb.open(
        args.output, map_size=1024 ** 4, readahead=False
    ) as env:
        pbar = tqdm(pool.imap(worker, speech_files), total=len(speech_files))

        mel_lengths = []
        text_lengths = []

        for i, record in enumerate(pbar):
            record_buffer = io.BytesIO()
            torch.save(record, record_buffer)

            with env.begin(write=True) as txn:
                txn.put(str(i).encode('utf-8'), record_buffer.getvalue())

            for char in record[1]:
                if char not in vocab:
                    vocab[char] = len(vocab) + N_META_CHAR

            mel_lengths.append(record[0].shape[0])
            text_lengths.append(len(record[1]))

            pbar.set_description(record[2])

        with env.begin(write=True) as txn:
            txn.put(b'length', str(len(speech_files)).encode('utf-8'))
            txn.put(
                b'meta',
                pickle.dumps(
                    {
                        'sr': PCM_SAMPLING_RATE,
                        'channels': PCM_CHANNELS,
                        'bit_depth': PCM_BIT_DEPTH,
                        'vocab': vocab,
                        'mel_lengths': mel_lengths,
                        'text_lengths': text_lengths,
                    }
                ),
            )
