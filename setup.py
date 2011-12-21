import os
from glob import glob
from setuptools import setup, find_packages

NAME = "nimfa"

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def get_package_data(topdir, excluded=set()):
    retval = []
    for dirname, subdirs, files in os.walk(os.path.join(NAME, topdir)):
        for x in excluded:
            if x in subdirs:
                subdirs.remove(x)
        retval.append(os.path.join(dirname[len(NAME)+1:], '*.*'))
    return retval

def get_data_files(dest, source):
    retval = []
    for dirname, subdirs, files in os.walk(source):
        retval.append(
            (os.path.join(dest, dirname[len(source)+1:]), glob(os.pathjoin(dirname, '*.*')))
        )
    return retval


setup(
    name = NAME,
    version = "1.0",
    author = "Marinka Zitnik",
    author_email = "marinka.zitnik@student.uni-lj.si", 
    description = "A Python Library for Nonnegative Matrix Factorization Techniques",
    url = "http://nimfa.biolab.si",
    download_url = "https://github.com/marinkaz/MF",
    packages = find_packages(),
    package_dir = {NAME: "./nimfa"},
    package_data = {NAME: get_package_data("datasets")},
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
