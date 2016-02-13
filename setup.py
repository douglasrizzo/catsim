#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='catsim',
    version='0.8',
    description='Computerized Adaptive Testing Simulator',
    author='Douglas De Rizzo Meneghetti',
    author_email='douglasrizzom@gmail.com',
    url='https://www.github.com/douglasrizzo/catsim',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 'scikit-learn'
    ],
    license='GPLv2'
)
