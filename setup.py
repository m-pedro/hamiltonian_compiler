
"""
Setup configuration for HamiltonianCompiler package
"""

from setuptools import setup, find_packages

setup(
    name="hamiltoniancompiler",
    version="1.0.0",
    author="Peter Morgan",
    author_email="pmorgan@deeplp.com",
    description="Efficient compilation of physical Hamiltonians for superconducting quantum computers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/m-pedro/hamiltonian_compiler",
    packages=find_packages(include=["hamiltoniancompiler", "hamiltoniancompiler.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
)
