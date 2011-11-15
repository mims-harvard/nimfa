import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "nimfa",
    version = "1.0",
    author = "Marinka Zitnik",
    author_email = "marinka.zitnik@student.uni-lj.si", 
    description = "A Python Library for Nonnegative Matrix Factorization Techniques",
    url = "http://nimfa.biolab.si",
    packages = find_packages(),
    package_dir = { "nimfa": "./nimfa"},
    license = "OSI Approved :: GNU General Public License (GPL)",
    long_description = read("README.rst"),
    classifiers = [
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering"
    ]
    )
