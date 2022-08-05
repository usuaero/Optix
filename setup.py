"""Optix: A Python optimization library."""

from setuptools import setup

setup(name = 'Optix',
    version = '2.1',
    description = 'Multivariable optimization tool with native multiprocessing.',
    url = 'https://github.com/usuaero/Optix',
    author = 'usuaero',
    author_email = 'doug.hunsaker@usu.edu',
    install_requires = ['numpy', 'multiprocessing_on_dill', 'pytest'],
    python_requires = '>=3.6.0',
    license = 'MIT',
    packages = ['optix'],
    zip_safe = False)
