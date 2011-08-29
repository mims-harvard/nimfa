
"""
    ####################################
    Orl_images (``examples.orl_images``)
    ####################################
    
    In this example we consider the image problem used in [Hoyer2004]_. 
    
    We used the ORL face database composed of 400 images of size 112 x 92. There are 40 persons, 10 images per
    each person. The images were taken at different times, lighting and facial expressions. The faces are in 
    an upright position in frontal view, with a slight left-right rotation. In example we performed factorization
    on reduced face images by constructing a matrix of shape 2576 (pixels) x 400 (faces) and on original face
    images by constructing a matrix of shape 10304 (pixels) x 400 (faces). 
    
    .. note:: ORL face images database used in this example is included in the `datasets` and need not to be
          downloaded. However, download links are listed in the ``datasets``. To run the example, the ORL face images
          must be find in the `ORL_faces` folder under `datasets`. 
          
    We used NMF - Euclidean, LSNMF and PSMF factorization methods to learn the basis images from the ORL database.  
          
    .. figure:: /images/orl_faces_500_iters_large_LSNMF.png
       :scale: 60 %
       :alt: Basis images of LSNMF obtained after 500 iterations on original face images. 
       :align: center

       Basis images of LSNMF obtained after 500 iterations on original face images. 



     .. figure:: /images/orl_faces_200_iters_small_NMF.png
       :scale: 60 %
       :alt: Basis images of NMF obtained after 200 iterations on reduced face images. 
       :align: center

       Basis images of NMF obtained after 200 iterations on reduced face images.   
       
       
    .. figure:: /images/orl_faces_200_iters_small_LSNMF.png
       :scale: 60 %
       :alt: Basis images of LSNMF obtained after 200 iterations on reduced face images.  
       :align: center

       Basis images of LSNMF obtained after 200 iterations on reduced face images.  
    
       
    .. figure:: /images/orl_faces_5_iters_small_PSMF_prior5.png
       :scale: 60 %
       :alt: Basis images of PSMF obtained after 5 iterations on reduced face images.  
       :align: center

       Basis images of PSMF obtained after 5 iterations on reduced face images.  
       
       
    To run the examples simply type::
        
        python orl_images.py
        
    or call the module's function::
    
        import mf.examples
        mf.examples.orl_images.run()
        
    .. note:: This example uses matplotlib library for producing a heatmap of a consensus matrix.
"""

import mf
import numpy as np
from matplotlib.pyplot import savefig, imshow, set_cmap
from os.path import dirname, abspath, sep

try:
    from PIL.Image import open, fromarray, new
    from PIL.ImageOps import expand
except ImportError, exc:
    raise SystemExit("PIL must be installed to run this example.")

def run():
    """Run LSNMF on ORL faces data set."""
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
    Perform LSNMF factorization on the ORL faces data matrix. 
    
    Return basis and mixture matrices of the fitted factorization model. 
    
    :param V: The ORL faces data matrix. 
    :type V: `numpy.matrix`
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
    Read face image data from ORL database. The matrix's shape is 2576 (pixels) x 400 (faces). 
    
    Step through each subject and each image. Reduce the size of the images by a factor of 0.5. 
    
    Return the ORL faces data matrix. 
    """
    print "Reading ORL faces database ..."
    dir = dirname(dirname(abspath(__file__)))+ sep + 'datasets' + sep + 'ORL_faces' + sep + 's'
    V = np.matrix(np.zeros((46 * 56, 400)))
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
    Preprocess ORL faces data matrix as Stan Li, et. al.
    
    :param V: The ORL faces data matrix. 
    :type V: `numpy.matrix`
    """
    print "Preprocessing data matrix ..." 
    min_val = V.min(axis = 0)
    V = V - np.mat(np.ones((V.shape[0], 1))) * min_val
    max_val = V.max(axis = 0) + 1e-4
    V = (255. * V) / (np.mat(np.ones((V.shape[0], 1))) * max_val)
    # avoid too large values 
    V = V / 100.
    print "... Finished."
    return V
            
def plot(W):
    """
    Plot basis vectors.
    
    :param W: Basis matrix of the fitted factorization model.
    :type W: `numpy.matrix`
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
    """Run the ORL faces example."""
    run()