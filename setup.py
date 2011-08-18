import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "MF",
    version = "0.1",
    author = "Marinka Zitnik",
    author_email = "marinka@zitnik.si", 
    description = "Python Matrix Factorization Techniques for Data Mining",
    url = "http://orange.biolab.si/trac/wiki/MatrixFactorization",
    packages = find_packages(),
    package_dir = { "mf": "./mf"},
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
