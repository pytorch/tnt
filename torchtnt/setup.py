#!/usr/bin/env python
from setuptools import find_packages, setup

VERSION = "0.0.5.1"

setup(
    # Metadata
    name="torchtnt",
    version=VERSION,
    author="PyTorch",
    author_email="daniellepintz@fb.com",
    url="https://github.com/pytorch/tnt/",
    description="A lightweight library for PyTorch training tools and utilities",
    license="BSD",
    # Package info
    packages=find_packages(exclude=("test", "docs")),
    zip_safe=True,
    install_requires=["torch", "six", "future", "visdom"],
)
