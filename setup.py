# WIP: This is a simple setup.py file for a Python package. It is used to install the package and its dependencies using pip.

from setuptools import setup, find_packages
from pybernetics import __version__, __author__, __email__, __description__, __url__, __license__

"""
Setup
=====

Note:
This file is subject to unnoted changes, as it is still in the development phase.

"""
setup(
    name="pybernetics",
    version=__version__,
    author=__author__,
    author_email=__email__,
    description=__description__,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url=__url__,
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",

        # Topics
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",

        # License
        "License :: OSI Approved :: MIT License",

        # Operating System
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
    ],
    python_requires=">=3.7",
)
