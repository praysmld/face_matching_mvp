"""
Setup script for backward compatibility.
Modern Python projects should use pyproject.toml for configuration.
"""

from setuptools import setup, find_packages

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
