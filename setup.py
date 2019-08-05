"""Optix: A Python, gradient-based optimization library."""

import os
from setuptools import setup

setup(name = 'Optix',
    version = '1.0',
    description = 'Continuous, multivariable optimization tool with native multiprocessing.',
    url = 'https://github.com/usuaero/Optix',
    author = 'usuaero',
    author_email = 'doug.hunsaker@usu.edu',
    install_requires = ['numpy'],
    python_requires = '>=3.6.0',
    license = 'MIT',
    packages = ['optix'],
    zip_safe = False)