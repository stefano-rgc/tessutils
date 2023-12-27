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
    install_requires=[
        'astropy>=5.1',
        'astroquery>=0.4.6',
        'joblib>=1.1.0',
        'lightkurve>=2.3.0',
        'matplotlib>=3.5.1',
        'numpy>=1.21.5',
        'pandas>=1.4.3',
        'peakutils>=1.3.4',
        'scipy>=1.7.3',
        'tqdm>=4.64.0',
    ],
    python_requires='>=3.10',
)
