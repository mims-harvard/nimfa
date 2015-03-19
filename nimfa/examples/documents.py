
"""
    ##################################
    Documents (``examples.documents``)
    ##################################

    In this example of text analysis we consider the text processing application inspired by [Albright2006]_.
    
    We used the Medlars data set, which is a collection of 1033 medical abstracts. For example we performed factorization
    on term-by-document matrix by constructing a matrix of shape 4765 (terms) x 1033 (documents). Original number
    of terms is 16017, the reduced number is a result of text preprocessing, namely removing stop words, too short words, 
    words that appear 2 times or less in the corpus and words that appear 50 times or more.

    .. note:: Medlars data set of medical abstracts used in this example is not included in the `datasets` and need to be
      downloaded. Download links are listed in the ``datasets``. Download compressed version of document text. To run the example, 
      the extracted Medlars data set must exist in the ``Medlars`` directory under ``datasets``. 
      
    Example of medical abstract::
        
        autolysis of bacillus subtilis by glucose depletion .                   
        in cultures in minimal medium, rapid lysis of cells of bacillus       
        subtilis was observed as soon as the carbon source, e.g. glucose, had   
        been completely consumed . the cells died and ultraviolet-absorbing     
        material was excreted in the medium . the results suggest that the cells
        lyse because of the presence of autolytic enzymes . in the presence of  
        glucose the damage to the cell wall caused by these enzymes is repaired 
        immediately . 
    
    Because of the nature of analysis, the resulting data matrix is very sparse. Therefore we use ``scipy.sparse`` matrix
    formats in factorization. This results in lower space consumption. Using, Standard NMF - Divergence, fitted
    factorization model is sparse as well, according to [Hoyer2004]_ measure of sparseness, the basis matrix has
    sparseness of 0.641 and the mixture matrix 0.863.
    
    .. note:: This sparseness 
              measure quantifies how much energy of a vector is packed into only few components. The sparseness of a vector
              is a real number in [0, 1]. Sparser vector has value closer to 1. The measure is 1 iff vector contains single
              nonzero component and the measure is equal to 0 iff all components are equal. Sparseness of a matrix is 
              the mean sparseness of its column vectors.
    
    The configuration of this example is sparse data matrix with Standard NMF - Divergence factorization method using 
    Random Vcol algorithm for initialization and rank 15 (the number of hidden topics). 
    
    Because of nonnegativity constraints, NMF has impressive benefits in terms of interpretation of its factors. In text
    processing applications, factorization rank can be considered the number of hidden topics present in the document
    collection. The basis matrix becomes a term-by-topic matrix whose columns are the basis vectors. Similar interpretation
    holds for the other factor, mixture matrix. Mixture matrix is a topic-by-document matrix with sparse nonnegative 
    columns. Element j of column 1 of mixture matrix measures the strength to which topic j appears in document 1. 
    
    .. figure:: /images/documents_basisW1.png
       :scale: 60 %
       :alt: Highest weighted terms in basis vector W1. 
       :align: center

       Interpretation of NMF - Divergence basis vectors on Medlars data set. Highest weighted terms in basis vector W1. The nonzero elements of column 1
       of W (W1), which is sparse and nonnegative, correspond to particular terms. By considering the highest weighted terms in this vector, 
       we can assign a label or topic to basis vector W1. As the NMF allows user the ability to interpret the basis vectors, a user might
       attach the label ``liver`` to basis vector W1. As a note, the term in 10th place, `viii`, is not a Roman numeral but
       instead `Factor viii`, an essential blood clotting factor also known as anti-hemophilic factor. It has been found
       to be synthesized and released into the bloodstream by the vascular, glomerular and tubular endothelium and 
       the sinusoidal cells of the ``liver``.
       
       
    .. figure:: /images/documents_basisW4.png
       :scale: 60 %
       :alt: Highest weighted terms in basis vector W4. 
       :align: center

       Interpretation of NMF basis vectors on Medlars data set. Highest weighted terms in basis vector W4. 
       
       
    .. figure:: /images/documents_basisW13.png
       :scale: 60 %
       :alt: Highest weighted terms in basis vector W13. 
       :align: center

       Interpretation of NMF basis vectors on Medlars data set. Highest weighted terms in basis vector W13. 
       
       
    .. figure:: /images/documents_basisW15.png
       :scale: 60 %
       :alt: Highest weighted terms in basis vector W15. 
       :align: center

       Interpretation of NMF basis vectors on Medlars data set. Highest weighted terms in basis vector W15. 
    
    To run the example simply type::
        
        python documents.py
        
    or call the module's function::
    
        import nimfa.examples
        nimfa.examples.documents.run()
        
    .. note:: This example uses ``matplotlib`` library for producing visual interpretation of NMF basis vectors on Medlars
              data set.
"""

from os.path import dirname, abspath
from os.path import join
from warnings import warn

import scipy.sparse as sp
import numpy as np

import nimfa

try:
    import matplotlib.pylab as plb
except ImportError as exc:
    warn("Matplotlib must be installed to run Documents example.")


def run():
    """Run NMF - Divergence on the Medlars data set."""
    V, term2idx, idx2term = read()
    V, term2idx, idx2term = preprocess(V, term2idx, idx2term)
    W, _ = factorize(V)
    plot(W, idx2term)


def factorize(V):
    """
    Perform NMF - Divergence factorization on the sparse Medlars data matrix. 
    
    Return basis and mixture matrices of the fitted factorization model. 
    
    :param V: The Medlars data matrix. 
    :type V: `scipy.sparse.csr_matrix`
    """
    nmf = nimfa.Nmf(V, seed="random_vcol", rank=12, max_iter=15, update="divergence",
                    objective="div")
    print("Algorithm: %s\nInitialization: %s\nRank: %d" % (nmf, nmf.seed, nmf.rank))
    fit = nmf()
    sparse_w, sparse_h = fit.fit.sparseness()
    print("""Stats:
            - iterations: %d
            - KL Divergence: %5.3f
            - Euclidean distance: %5.3f
            - Sparseness basis: %5.3f, mixture: %5.3f""" % (fit.fit.n_iter, fit.distance(), fit.distance(metric='euclidean'), sparse_w, sparse_h))
    return fit.basis(), fit.coef()


def read():
    """
    Read medical abstracts data from Medlars data set. 
    
    Construct a term-by-document matrix. This matrix is sparse, therefore ``scipy.sparse`` format is used. For construction
    LIL sparse format is used, which is an efficient structure for constructing sparse matrices incrementally. 
    
    Return the Medlars sparse data matrix in LIL format, term-to-index `dict` translator and index-to-term 
    `dict` translator. 
    """
    print("Read Medlars medical abstracts data set")
    dir = join(dirname(dirname(abspath(__file__))), "datasets", "Medlars", "med.all")
    doc = open(dir)
    V = sp.lil_matrix((16017, 1033))
    term2idx = {}
    idx2term = {}
    n_free = 0
    line = doc.readline()
    for abstract in range(1033):
        ii = int(line.split()[1])
        # omit .W char
        doc.readline()
        line = doc.readline()
        while line != ".I " + str(ii + 1) and line != "":
            for term in line.split():
                term = term.strip().replace(',', '').replace('.', '')
                if term not in term2idx:
                    term2idx[term] = n_free
                    idx2term[n_free] = term
                    n_free += 1
                V[term2idx[term], ii - 1] += 1
            line = doc.readline().strip()
    return V, term2idx, idx2term


def preprocess(V, term2idx, idx2term):
    """
    Preprocess Medlars data matrix. Remove stop words, digits, too short words, words that appear 2 times or less 
    in the corpus and words that appear 50 times or more.
    
    Return preprocessed term-by-document sparse matrix in CSR format. Returned matrix's shape is 4765 (terms) x 1033 (documents). 
    The sparse data matrix is converted to CSR format for fast arithmetic and matrix vector operations. Return
    updated index-to-term and term-to-index translators.
    
    :param V: The Medlars data matrix. 
    :type V: `scipy.sparse.lil_matrix`
    :param term2idx: Term-to-index translator.
    :type term2idx: `dict`
    :param idx2term: Index-to-term translator.
    :type idx2term: `dict`
    """
    print("Data preprocessing")
    # remove stop words, digits, too short words
    rem = set()
    for term in term2idx:
        if term in stop_words or len(term) <= 2 or str.isdigit(term):
            rem.add(term2idx[term])
    # remove words that appear two times or less in corpus
    V = V.tocsr()
    for r in range(V.shape[0]):
        if V[r, :].sum() <= 2 or V[r,:].sum() >= 50:
            rem.add(r)
    retain = set(range(V.shape[0])).difference(rem)
    n_free = 0
    V1 = sp.lil_matrix((V.shape[0] - len(rem), 1033))
    for r in retain:
        term2idx[idx2term[r]] = n_free
        idx2term[n_free] = idx2term[r]
        V1[n_free, :] = V[r,:] 
        n_free += 1
    return V1.tocsr(), term2idx, idx2term


def plot(W, idx2term):
    """
    Plot the interpretation of NMF basis vectors on Medlars data set. 
    
    :param W: Basis matrix of the fitted factorization model.
    :type W: `scipy.sparse.csr_matrix`
    :param idx2term: Index-to-term translator.
    :type idx2term: `dict`
    """
    print("Plot highest weighted terms in basis vectors")
    for c in range(W.shape[1]):
        if sp.isspmatrix(W):
            top10 = np.argsort(np.asarray(W[:, c].todense()).flatten())[-10:]
            val = W[top10, c].todense()
        else:
            top10 = np.argsort(np.asarray(W[:, c]).flatten())[-10:]
            val = W[top10, c]
        plb.figure(c + 1)
        plb.barh(np.arange(10) + .5, val, color="yellow", align="center")
        plb.yticks(np.arange(10) + .5, [idx2term[idx] for idx in top10][::-1])
        plb.xlabel("Weight")
        plb.ylabel("Term")
        plb.title("Highest Weighted Terms in Basis Vector W%d" % (c + 1))
        plb.grid(True)
        plb.savefig("documents_basisW%d.png" % (c + 1), bbox_inches="tight")


stop_words = [
    "a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot",
    "could", "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from",
    "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however",
    "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", "like", "likely",
    "may", "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off",
    "often", "on", "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she",
    "should", "since", "so", "some", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would",
    "yet", "you", "your", ".", " ", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "during",
    "changes", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)", "(8)", "(9)", "usually", "involved",
    "labeled"]


if __name__ == "__main__":
    """Run the Medlars example."""
    run()
