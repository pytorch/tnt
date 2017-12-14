#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages

VERSION = '0.0.2'

long_description = "an abstraction to train neural networks"

setup_info = dict(
    # Metadata
    name='torchnet',
    version=VERSION,
    author='Barry Kui',
    author_email='xukui.cs@gmail.com',
    url='https://github.com/barrykui/tnt',
    description='an abstraction to train neural networks',
    long_description=long_description,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,

    install_requires=[
        'torch',
        'six',
        'visdom'
    ]
)

setup(**setup_info)
