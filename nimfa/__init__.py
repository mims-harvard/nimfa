
"""
#################
Nimfa (``nimfa``)
#################

Nimfa is a Python module which includes a number of published matrix
factorization algorithms, initialization methods, quality and performance
measures and facilitates the combination of these to produce new strategies.
The library represents a unified and efficient interface to matrix
factorization algorithms and methods.
"""

__license__ = 'BSD'
__version__ = '1.3.2'
__maintainer__ = 'Marinka Zitnik'
__email__ = 'marinka@cs.stanford.edu'


from nimfa import examples
from nimfa.methods.factorization import *
from .version import \
    short_version as __version__, git_revision as __git_version__
