from setuptools import setup, find_packages

setup(
    name='tessutils',
    version='1.0.0',
    author='Stefano Garcia',
    author_email='stefano.rgc@gmail.com',
    description='Pick a TIC number and obtain a reduced light curve for all TESS sectors available along with diagnostic plots about the reduction.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
)
