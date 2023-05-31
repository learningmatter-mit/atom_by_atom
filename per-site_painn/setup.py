import os
import io

from setuptools import setup, find_packages


def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()


setup(
    name="persite_painn",
    version="0.1.0",
    author="Hoje Chun",
    email="hoje.chun316@gmail.com",
    python_requires=">=3.9",
    packages=find_packages(
        ".",
        exclude=[
            "data_cache",
            "data_cache.*",
            "data_raw",
            "data_raw.*",
            "notebooks",
            "notebooks.*",
            "results",
            "results.*",
        ],
    ),
    description="Per-site PaiNN",
    long_description=read("README.md"),
)
