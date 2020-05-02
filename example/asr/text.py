import re

re_whitespace = re.compile(r'\s+')


def collapse_whitespace(text):
    return re_whitespace.sub(' ', text)
