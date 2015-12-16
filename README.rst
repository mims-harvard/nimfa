Nimfa
-----

Nimfa is a Python module that implements many algorithms for nonnegative matrix factorization.

Documentation and examples are at `Nimfa website`_.

.. _Nimfa website: http://nimfa.biolab.si

****

`Hidden patients and hidden genes - Understanding cancer data with matrix factorization`_ is
a tutorial-like IPython notebook on how one can use Nimfa to analyze breast cancer transcriptome data sets from The
International Cancer Genome Consortium (`ICGC`_). A column about the analysis of cancer data using Nimfa recently
appearead in the `ACM XRDS magazine`_.

.. _Hidden patients and hidden genes - Understanding cancer data with matrix factorization: http://nbviewer.ipython.org/github/marinkaz/nimfa-ipynb/blob/master/ICGC%20and%20Nimfa.ipynb
.. _ICGC: https://dcc.icgc.org
.. _ACM XRDS magazine: http://dl.acm.org/citation.cfm?id=2809623.2788526&coll=portal&dl=ACM

Usage
-----

Run alternating least squares nonnegative matrix factorization with projected gradients and Random Vcol initialization algorithm on medulloblastoma gene expression data::

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


Citing
------

::

    @article{Zitnik2012,
      title     = {Nimfa: A Python Library for Nonnegative Matrix Factorization},
      author    = {{\v{Z}}itnik, Marinka and Zupan, Bla{\v{z}}},
      journal   = {Journal of Machine Learning Research},
      volume    = {13},
      pages     = {849-853},
      year      = {2012}
    }


License
-------

nimfa - A Python Library for Nonnegative Matrix Factorization Techniques
Copyright (C) 2011-2015 Marinka Zitnik and Blaz Zupan

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.


JMLR Warranty
-------------

THIS SOURCE CODE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND, AND ITS AUTHOR AND THE JOURNAL OF MACHINE LEARNING RESEARCH (JMLR) 
AND JMLR'S PUBLISHERS AND DISTRIBUTORS, DISCLAIM ANY AND ALL WARRANTIES, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF 
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, AND ANY WARRANTIES OR NON INFRINGEMENT. THE USER ASSUMES ALL LIABILITY 
AND RESPONSIBILITY FOR USE OF THIS SOURCE CODE, AND NEITHER THE AUTHOR NOR JMLR, NOR JMLR'S PUBLISHERS AND DISTRIBUTORS, WILL BE 
LIABLE FOR DAMAGES OF ANY KIND RESULTING FROM ITS USE. 

Without limiting the generality of the foregoing, neither the author, nor JMLR, nor JMLR's publishers and distributors, warrant that 
the Source Code will be error-free, will operate without interruption, or will meet the needs of the user.






