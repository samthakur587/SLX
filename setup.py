# setup.py
from setuptools import setup, find_packages

setup(
    name='slx',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
    ],
)   



if __name__ == '__main__':
    setup()
