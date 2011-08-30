
"""
    ##################################
    Documents (``examples.documents``)
    ##################################


    .. note:: Medlars data set of medical abstracts used in this example is not included in the `datasets` and need to be
      downloaded. Download links are listed in the ``datasets``. Download compressed version of document text. To run the example, 
      the extracted Medlars data set must be find in the ``Medlars`` folder under ``datasets``. 
    
    
    To run the examples simply type::
        
        python documents.py
        
    or call the module's function::
    
        import mf.examples
        mf.examples.documents.run()
        
    .. note:: This example uses matplotlib library for producing visual interpretation of NMF basis vectors on Medlars
              data set.
"""

import mf
import numpy as np
import scipy.sparse as sp
from matplotlib.pyplot import savefig, imshow, set_cmap
from os.path import dirname, abspath, sep

def run():
    """Run NMF - Divergence on the Medlars data set."""
    # read medical abstracts from Medlars data set 
    V = read()
    # preprocess Medlars data matrix
    V = preprocess(V)
    # run factorization
    W, _ = factorize(V)
    # plot interpretation of NMF basis vectors on Medlars data set. 
    plot(W)
    
def factorize(V):
    """
    Perform NMF - Divergence factorization on the sparse Medlars data matrix. 
    
    Return basis and mixture matrices of the fitted factorization model. 
    
    :param V: The Medlars data matrix. 
    :type V: `scipy.sparse.csr_matrix`
    """
    print "Performing LSNMF factorization ..." 
    model = mf.mf(V, 
                  seed = "random_vcol",
                  rank = 25, 
                  method = "lsnmf", 
                  max_iter = 50,
                  initialize_only = True,
                  sub_iter = 10,
                  inner_sub_iter = 10, 
                  beta = 0.1,
                  min_residuals = 1e-8)
    fit = mf.mf_run(model)
    print " ... Finished"
    print """Stats:
            - iterations: %d
            - final projected gradients norm: %5.3f
            - Euclidean distance: %5.3f""" % (fit.fit.n_iter, fit.distance(), fit.distance(metric = 'euclidean'))
    return fit.basis(), fit.coef()
    
def read():
    """
    Read medical abstracts data from Medlars data set. The matrix's shape is 5831 (terms) x 1033 (documents). 
    
    Construct term-by-document matrix. This matrix is sparse, therefore ``scipy.sparse`` format is used. For construction
    LIL sparse format is used, which is an efficient structure for constructing sparse matrices incrementally. 
    
    Return the Medlars sparse data matrix. 
    """
    print "Reading Medlars medical abstracts data set ..."
    dir = dirname(dirname(abspath(__file__)))+ sep + 'datasets' + sep + 'Medlars' + sep + 'med.all'
    V = sp.lil_matrix((5831, 1033))
    for subject in xrange(40):
        for image in xrange(10):
            im = open(dir + str(subject + 1) + sep + str(image + 1) + ".pgm")
            # reduce the size of the image
            im = im.resize((46, 56))
            V[:, image * subject + image] = np.mat(np.asarray(im).flatten()).T      
    print "... Finished."
    return V
            
def preprocess(V):
    """
    Preprocess Medlars data matrix.
    
    Return preprocessed term-by-document matrix. The sparse data matrixis converted to CSR format for fast arithmetic
    and matrix vector operations. 
    
    :param V: The Medlars data matrix. 
    :type V: `scipy.sparse.lil_matrix`
    """
    print "Preprocessing data matrix ..." 
    min_val = V.min(axis = 0)
    V = V - np.mat(np.ones((V.shape[0], 1))) * min_val
    max_val = V.max(axis = 0) + 1e-4
    V = (255. * V) / (np.mat(np.ones((V.shape[0], 1))) * max_val)
    # avoid too large values 
    V = V / 100.
    print "... Finished."
    return V.tocsr()
            
def plot(W):
    """
    Plot the interpretation of NMF basis vectors on Medlars data set. 
    
    :param W: Basis matrix of the fitted factorization model.
    :type W: `scipy.sparse.csr_matrix`
    """
    set_cmap('gray')
    blank = new("L", (225 + 6, 280 + 6))
    for i in xrange(5):
        for j in xrange(5):
            basis = np.array(W[:, 5 * i + j])[:, 0].reshape((56, 46))
            basis = basis / np.max(basis) * 255
            basis = 255 - basis
            ima = fromarray(basis)
            expand(ima, border = 1, fill = 'black')
            blank.paste(ima.copy(), (j * 46 + j, i * 56 + i))
    imshow(blank)
    savefig("orl_faces.png")

if __name__ == "__main__":
    """Run the Medlars example."""
    run()
