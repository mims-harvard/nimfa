import os
from glob import glob
from setuptools import setup, find_packages
import subprocess
import imp


DISTNAME = 'nimfa'
MAINTAINER = 'Marinka Zitnik'
MAINTAINER_EMAIL = 'marinka@cs.stanford.edu'
DESCRIPTION = 'A Python module for nonnegative matrix factorization'
LONG_DESCRIPTION = open('README.md').read()
URL = 'http://nimfa.biolab.si'
DOWNLOAD_URL = 'http://github.com/marinkaz/nimfa'
KEYWORDS = ['matrix factorization', 'nonnegative matrix factorization',
            'bioinformatics', 'data mining', 'machine learning']
LICENSE = 'BSD'
VERSION = '1.3.2'
ISRELEASED = True

INSTALL_REQUIRES = (
    'numpy>=1.7.0',
    'scipy>=0.12.0',
)


def get_package_data(topdir, excluded=set()):
    retval = []
    for dirname, subdirs, files in os.walk(os.path.join(DISTNAME, topdir)):
        for x in excluded:
            if x in subdirs:
                subdirs.remove(x)
        retval.append(os.path.join(dirname[len(DISTNAME)+1:], '*.*'))
    return retval


def get_data_files(dest, source):
    retval = []
    for dirname, subdirs, files in os.walk(source):
        retval.append(
            (os.path.join(dest, dirname[len(source)+1:]), glob(os.pathjoin(dirname, '*.*')))
        )
    return retval


# Return the git revision as a string
def git_version():
    """Return the git revision as a string.

    Copied from numpy setup.py
    """
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"
    return GIT_REVISION


def write_version_py(filename='nimfa/version.py'):
    # Copied from numpy setup.py
    cnt = """
# THIS FILE IS GENERATED FROM NIMFA SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
    short_version += ".dev"
"""
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('nimfa/version.py'):
        # must be a source distribution, use existing version file
        version = imp.load_source('nimfa.version', 'nimfa/version.py')
        GIT_REVISION = version.git_revision
    else:
        GIT_REVISION = 'Unknown'

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()



def setup_package():
    write_version_py()
    setup(
        name=DISTNAME,
        version=VERSION,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        url=URL,
        download_url=DOWNLOAD_URL,
        keywords=KEYWORDS,
        install_requires=INSTALL_REQUIRES,
        packages=find_packages(),
        package_dir={DISTNAME: './nimfa'},
        package_data={DISTNAME: get_package_data('datasets')},
        license=LICENSE,
        long_description=LONG_DESCRIPTION,
        classifiers=['Intended Audience :: Science/Research',
                     'Intended Audience :: Developers',
                     'License :: OSI Approved',
                     'Programming Language :: Python',
                     'Topic :: Software Development',
                     'Topic :: Scientific/Engineering',
                     'Topic :: Scientific/Engineering :: Artificial Intelligence',
                     'Topic :: Scientific/Engineering :: Bio-Informatics',
                     'Programming Language :: Python :: 2',
                     'Programming Language :: Python :: 3',],
        )


if __name__ == "__main__":
    setup_package()
