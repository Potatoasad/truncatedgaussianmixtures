# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='truncatedgaussianmixtures',
    version='0.1.3',
    description="Fit gaussian mixture models using truncated gaussian kernels. This is a python wrapper around the julia package TruncatedGaussianMixtures.jl",
    author='Asad Hussain',
    author_email='asadh@utexas.edu',
    url='https://github.com/potatoasad/truncatedgaussianmixtures',
    packages=find_packages(exclude=('tests', 'docs', 'dev'))
)
