import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "MF - Matrix Factorization Techniques for Data Mining",
    version = "1.0",
    author = "Marinka Zitnik",
    author_email = "marinka@zitnik.si", 
    description = "Python Matrix Factorization Techniques for Data Mining",
    url = "http://helikoid.si/mf/index.html",
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
