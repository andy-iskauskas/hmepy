import os
import runpy
from setuptools import setup, find_packages

# Get version
cwd = os.path.abspath(os.path.dirname(__file__))
versionpath = os.path.join(cwd, 'hmepy', 'version.py')
version = runpy.run_path(versionpath)['__version__']

# Get the documentation
# with open(os.path.join(cwd, 'README.rst'), "r") as f:
#     long_description = f.read()

CLASSIFIERS = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: MIT",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Development Status :: 5 - Production/Stable",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]

setup(
    name="hmepy",
    version=version,
    author="Andrew Iskauskas, TJ McKinley",
    author_email="andrew.iskauskas@durham.ac.uk",
    description="HMEpy: History Matching and Emulation for Python",
    # long_description=long_description,
    # long_description_content_type="text/x-rst",
    url='https://hmer-package.github.io/website/',
    keywords=["emulation", "simulation", "history matching", "calibration"],
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas>=2.2.2', 
        'matplotlib',
        'copy',
        'json',
    ],
)