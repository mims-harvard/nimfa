Nimfa
-----

[![build: passing](https://img.shields.io/travis/marinkaz/nimfa.svg)](https://travis-ci.org/marinkaz/nimfa)
[![build: passing](https://coveralls.io/repos/marinkaz/nimfa/badge.svg)](https://coveralls.io/github/marinkaz/nimfa?branch=master)
[![GitHub release](https://img.shields.io/github/release/marinkaz/nimfa.svg)](https://GitHub.com/marinkaz/nimfa/releases/)
[![BSD license](https://img.shields.io/badge/License-BSD-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Nimfa is a Python module that implements many algorithms for nonnegative matrix factorization. Nimfa is distributed under the BSD license.

The project was started in 2011 by Marinka Zitnik as a Google Summer of Code project, and since
then many volunteers have contributed. See AUTHORS file for a complete list of contributors.

It is currently maintained by a team of volunteers.

[**[News:]**](https://github.com/marinkaz/scikit-fusion) [Scikit-fusion](https://github.com/marinkaz/scikit-fusion), collective latent factor models, matrix factorization for data fusion and learning over heterogeneous data.

[**[News:]**](https://github.com/mims-harvard/fastGNMF) [fastGNMF](https://github.com/mims-harvard/fastGNMF), fast implementation of graph-regularized non-negative matrix factorization using [Facebook FAISS](https://github.com/facebookresearch/faiss).

Important links
---------------

- Official source code repo: https://github.com/marinkaz/nimfa
- HTML documentation (stable release): http://ai.stanford.edu/~marinka/nimfa
- Download releases: http://github.com/marinkaz/nimfa/releases
- Issue tracker: http://github.com/marinkaz/nimfa/issues

Dependencies
------------

Nimfa is tested to work under Python 2.7 and Python 3.4.

The required dependencies to build the software are NumPy >= 1.7.0,
SciPy >= 0.12.0.

For running the examples Matplotlib >= 1.1.1 is required.

Install
-------

This package uses setuptools, which is a common way of installing
python modules. To install in your home directory, use:

    python setup.py install --user

To install for all users on Unix/Linux:
    
    sudo python setup.py install

For more detailed installation instructions,
see the web page http://ai.stanford.edu/~marinka/nimfa.

Use
---

Run alternating least squares nonnegative matrix factorization with projected gradients and Random Vcol initialization algorithm on medulloblastoma gene expression data:

    >>> import nimfa
    >>> V = nimfa.examples.medulloblastoma.read(normalize=True)
    >>> lsnmf = nimfa.Lsnmf(V, seed='random_vcol', rank=50, max_iter=100)
    >>> lsnmf_fit = lsnmf()
    >>> print('Rss: %5.4f' % lsnmf_fit.fit.rss())
    Rss: 0.2668
    >>> print('Evar: %5.4f' % lsnmf_fit.fit.evar())
    Evar: 0.9997
    >>> print('K-L divergence: %5.4f' % lsnmf_fit.distance(metric='kl'))
    K-L divergence: 38.8744
    >>> print('Sparseness, W: %5.4f, H: %5.4f' % lsnmf_fit.fit.sparseness())
    Sparseness, W: 0.7297, H: 0.8796


Cite
----

    @article{Zitnik2012,
      title     = {Nimfa: A Python Library for Nonnegative Matrix Factorization},
      author    = {Zitnik, Marinka and Zupan, Blaz},
      journal   = {Journal of Machine Learning Research},
      volume    = {13},
      pages     = {849-853},
      year      = {2012}
    }

Selected publications (Methods)
------------------------------

- Data fusion by matrix factorization: http://dx.doi.org/10.1109/TPAMI.2014.2343973
- Jumping across biomedical contexts using compressive data fusion: https://academic.oup.com/bioinformatics/article/32/12/i90/2240593
- Survival regression by data fusion: http://www.tandfonline.com/doi/abs/10.1080/21628130.2015.1016702
- Gene network inference by fusing data from diverse distributions: https://academic.oup.com/bioinformatics/article/31/12/i230/216398
- Fast optimization of non-negative matrix tri-factorization: https://doi.org/10.1371/journal.pone.0217994

Selected publications (Applications)
------------------------------------

- A comprehensive structural, biochemical and biological profiling of the human NUDIX hydrolase family: https://www.nature.com/articles/s41467-017-01642-w
- Gene prioritization by compressive data fusion and chaining: http://dx.doi.org/10.1371/journal.pcbi.1004552
- Discovering disease-disease associations by fusing systems-level molecular data: http://www.nature.com/srep/2013/131115/srep03202/full/srep03202.html
- Matrix factorization-based data fusion for gene function prediction in baker's yeast and slime mold: http://www.worldscientific.com/doi/pdf/10.1142/9789814583220_0038
- Matrix factorization-based data fusion for drug-induced liver injury prediction: http://www.tandfonline.com/doi/abs/10.4161/sysb.29072
- Collective pairwise classification for multi-way analysis of disease and drug data: https://doi.org/10.1142/9789814749411_0008

Tutorials
---------

- Hidden Genes: Understanding cancer data with matrix factorization, ACM XRDS: Crossroads: https://dl.acm.org/citation.cfm?id=2809623.2788526 [[Jupyter Notebook]](https://nbviewer.jupyter.org/github/marinkaz/nimfa-ipynb/blob/master/ICGC%20and%20Nimfa.ipynb)

<p align="center">
<img src="https://github.com/marinkaz/nimfa/blob/master/tutorial-diseases.png" width="800" align="center">
</p>
