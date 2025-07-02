"""
Entry point for installing the package.
"""

# Native Library | setuptools
from setuptools import setup, find_packages

# (1): Use setup() to... set up the package:
setup(
    name = "bkm10_lib",
    version = "1.2.5",
    packages=find_packages(),
)
