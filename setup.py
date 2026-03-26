"""
Allows installing src directory as Python package.
"""
from setuptools import setup

setup(
    name="cancer_ml",  # The name of your package
    version="0.1.0",         # Version number
    description="Segment some cancer",  # A short description
    packages=["cancer_ml"],  # List of package directories
)