# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

setup(
    name='truncatedgaussianmixtures',
    version='0.1.0',
    description='''Fit gaussian mixture models using truncated gaussian kernels. This is a python wrapper around the julia package TruncatedGaussianMixtures.jl''',
    long_description=readme,
    author='Asad Hussain',
    author_email='asadh@utexas.edu',
    url='https://github.com/potatoasad/truncatedgaussianmixtures',
    packages=find_packages(exclude=('tests', 'docs', 'dev'))
)