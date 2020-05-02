from setuptools import setup, find_packages

setup(
    name="torch_imputer",
    version="0.1.0",
    description="Implementation of Imputer: Sequence Modelling via Imputation and Dynamic Programming in PyTorch ",
    url="https://github.com/rosinality/imputer-pytorch",
    author="Kim Seonghyeon",
    author_email="kim.seonghyeon@navercorp.com",
    license="MIT",
    python_requires=">=3.6",
    packages=find_packages(exclude=["example"]),
)
