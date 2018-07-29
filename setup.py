#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages

VERSION = '0.0.3.1'

long_description = "Simple tools for logging and visualizing, loading and training"

setup_info = dict(
    # Metadata
    name='torchnet',
    version=VERSION,
    author='PyTorch',
    author_email='sergey.zagoruyko@enpc.fr',
    url='https://github.com/pytorch/tnt/',
    description='an abstraction to train neural networks',
    long_description=long_description,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test', 'docs')),

    zip_safe=True,

    install_requires=[
        'torch',
        'six',
        'visdom'
    ]
)

setup(**setup_info)
