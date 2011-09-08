
"""
    ######################################
    Cbcl_images (``examples.cbcl_images``)
    ######################################
    
    .. note:: The CBCL face images database used in this example is not included in the `datasets`. If you wish to
              perform the CBCL data experiments, start by downloading the images.  Download links are listed in the 
              ``datasets``. To run the example, uncompress the data and put it into corresponding data directory, namely 
              the extracted CBCL data set must be find in the ``CBCL_faces`` directory under ``datasets``. Once you have 
              the data installed, you are ready to start running the experiments. 
      
    .. figure:: /images/cbcl_faces_50_iters_LSNMF.png
       :scale: 70 %
       :alt: Basis images of LSNMF obtained after 50 iterations on original CBCL face images. 
       :align: center
       
       Basis images of LSNMF obtained after 50 iterations on original CBCL face images.
       
       
    .. figure:: /images/cbcl_faces_50_iters_NMF.png
       :scale: 70 %
       :alt: Basis images of NMF obtained after 50 iterations on original CBCL face images. 
       :align: center
       
       Basis images of NMF obtained after 50 iterations on original CBCL face images.
       
        
    .. figure:: /images/cbcl_faces_10_iters_SNMF_L.png
       :scale: 70 %
       :alt: Basis images of LSNMF obtained after 10 iterations on original CBCL face images. 
       :align: center
       
       Basis images of SNMF/L obtained after 10 iterations on original CBCL face images.
       
       
    .. figure:: /images/cbcl_faces_10_iters_SNMF_R.png
       :scale: 70 %
       :alt: Basis images of SNMF/R obtained after 10 iterations on original CBCL face images. 
       :align: center
       
       Basis images of SNMF/R obtained after 10 iterations on original CBCL face images.
       
          
    To run the example simply type::
        
        python cbcl_images.py
        
    or call the module's function::
    
        import mf.examples
        mf.examples.cbcl_images.run()
        
    .. note:: This example uses matplotlib library for producing visual interpretation of basis vectors. It uses PIL 
              library for displaying face images. 
    
"""

import mf
import numpy as np
from os.path import dirname, abspath, sep

try:
    from matplotlib.pyplot import savefig, imshow, set_cmap
except ImportError, exc:
    raise SystemExit("Matplotlib must be installed to run this example.")

try:
    from PIL.Image import open, fromarray, new
    from PIL.ImageOps import expand
except ImportError, exc:
    raise SystemExit("PIL must be installed to run this example.")

def run():
    """Run LSNMF on CBCL faces data set."""
    # read face image data from ORL database 
    V = read()
    # preprocess ORL faces data matrix
    V = preprocess(V)
    # run factorization
    W, _ = factorize(V)
    # plot parts-based representation 
    plot(W)
    
def factorize(V):
    """
    Perform LSNMF factorization on the CBCL faces data matrix. 
    
    Return basis and mixture matrices of the fitted factorization model. 
    
    :param V: The CBCL faces data matrix. 
    :type V: `numpy.matrix`
    """
    print "Performing LSNMF factorization ..." 
    model = mf.mf(V, 
                  seed = "random_vcol",
                  rank = 49, 
                  method = "lsnmf", 
                  max_iter = 50, 
                  initialize_only = True,
                  sub_iter = 10,
                  inner_sub_iter = 10, 
                  beta = 0.1,
                  min_residuals = 1e-8)
    fit = mf.mf_run(model)
    print "... Finished"
    print """Stats:
            - iterations: %d
            - final projected gradients norm: %5.3f
            - Euclidean distance: %5.3f""" % (fit.fit.n_iter, fit.distance(), fit.distance(metric = 'euclidean'))
    return fit.basis(), fit.coef()
    
def read():
    """
    Read face image data from the CBCL database. The matrix's shape is 361 (pixels) x 2429 (faces). 
    
    Step through each subject and each image. Images' sizes are not reduced.  
    
    Return the CBCL faces data matrix. 
    """
    print "Reading CBCL faces database ..."
    dir = dirname(dirname(abspath(__file__)))+ sep + 'datasets' + sep + 'CBCL_faces' + sep + 'face'
    V = np.matrix(np.zeros((19 * 19, 2429)))
    for image in xrange(2429):
        im = open(dir + sep + "face0" + str(image + 1).zfill(4) + ".pgm")
        V[:, image] = np.mat(np.asarray(im).flatten()).T      
    print "... Finished."
    return V
            
def preprocess(V):
    """
    Preprocess CBCL faces data matrix as Lee and Seung.
    
    Return normalized and preprocessed data matrix. 
    
    :param V: The CBCL faces data matrix. 
    :type V: `numpy.matrix`
    """
    print "Preprocessing data matrix ..."
    V = V - V.mean()
    V = V / np.sqrt(np.multiply(V, V).mean())
    V = V + 0.25
    V = V * 0.25
    V = np.minimum(V, 1)
    V = np.maximum(V, 0)
    print "... Finished."
    return V
            
def plot(W):
    """
    Plot basis vectors.
    
    :param W: Basis matrix of the fitted factorization model.
    :type W: `numpy.matrix`
    """
    set_cmap('gray')
    blank = new("L", (133 + 6, 133 + 6))
    for i in xrange(7):
        for j in xrange(7):
            basis = np.array(W[:, 7 * i + j])[:, 0].reshape((19, 19))
            basis = basis / np.max(basis) * 255
            basis = 255 - basis
            ima = fromarray(basis)
            ima = ima.rotate(180)
            expand(ima, border = 1, fill = 'black')
            blank.paste(ima.copy(), (j * 19 + j, i * 19 + i))
    imshow(blank)
    savefig("cbcl_faces.png")

if __name__ == "__main__":
    """Run the CBCL faces example."""
    run()