# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='truncatedgaussianmixtures',
    version='0.1.7',
    description="Fit gaussian mixture models using truncated gaussian kernels. This is a python wrapper around the julia package TruncatedGaussianMixtures.jl",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Asad Hussain',
    author_email='asadh@utexas.edu',
    url='https://github.com/potatoasad/truncatedgaussianmixtures',
    packages=find_packages(exclude=('tests', 'docs', 'dev'))
)
