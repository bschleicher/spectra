from setuptools import setup, find_packages

setup(
    name='blockspec',
    version="0.1",
    description='A module for spectral block analyses for the fact telescope',
    url='https://github.com/mblnk/spectra',
    author='Michael Blank',
    author_email='michael.blank@stud-mail.uni-wuerzburg.de',
    packages=find_packages(),
    install_requires=[
        'pyfact',
        'tqdm',
    ]
)
