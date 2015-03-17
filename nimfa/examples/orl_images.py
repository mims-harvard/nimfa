
"""
    ####################################
    ORL Images (``examples.orl_images``)
    ####################################
    
    In this example of image processing we consider the image problem presented in [Hoyer2004]_. 
    
    We used the ORL face database composed of 400 images of size 112 x 92. There are 40 persons, 10 images per
    each person. The images were taken at different times, lighting and facial expressions. The faces are in 
    an upright position in frontal view, with a slight left-right rotation. In example we performed factorization
    on reduced face images by constructing a matrix of shape 2576 (pixels) x 400 (faces) and on original face
    images by constructing a matrix of shape 10304 (pixels) x 400 (faces). To avoid too large values, the data matrix is 
    divided by 100. Indeed, this division does not has any major impact on performance of the MF methods. 
    
    .. note:: The ORL face images database used in this example is included in the `datasets` and does not need to be
          downloaded. However, download links are listed in the ``datasets``. To run the example, the ORL face images
          must exist in the ``ORL_faces`` directory under ``datasets``. 
          
    We experimented with the Standard NMF - Euclidean, LSNMF and PSMF factorization methods to learn the basis images from the ORL database. The
    number of bases is 25. In [Lee1999]_ Lee and Seung showed that Standard NMF (Euclidean or divergence) found a parts-based
    representation when trained on face images from CBCL database. However, applying NMF to the ORL data set, in which images
    are not as well aligned, a global decomposition emerges. To compare, this example applies different MF methods to the face 
    images data set. Applying MF methods with sparseness constraint, namely PSMF, the resulting bases are not global, but instead
    give spatially localized representations, as can be seen from the figure. Similar conclusions are published in [Hoyer2004]_.
    Setting a high sparseness value for the basis images results in a local representation. 
    
    
    .. note:: It is worth noting that sparseness constraints do not always lead to local solutions. Global solutions can 
              be obtained by forcing low sparseness on basis matrix and high sparseness on coefficient matrix - forcing 
              each coefficient to represent as much of the image as possible. 
          
          
    .. figure:: /images/orl_faces_500_iters_large_LSNMF.png
       :scale: 70 %
       :alt: Basis images of LSNMF obtained after 500 iterations on original face images. 
       :align: center

       Basis images of LSNMF obtained after 500 iterations on original face images. The bases trained by LSNMF are additive
       but not spatially localized for representation of faces. Random VCol initialization algorithm is used. The number of
       subiterations for solving subproblems in LSNMF is a important issues. However, we stick to default and use 10 subiterations
       in this example. 


    .. figure:: /images/orl_faces_200_iters_small_NMF.png
       :scale: 70 %
       :alt: Basis images of NMF - Euclidean obtained after 200 iterations on reduced face images. 
       :align: center

       Basis images of NMF - Euclidean obtained after 200 iterations on reduced face images. The images show that
       the bases trained by NMF are additive but not spatially localized for representation of faces. The Euclidean
       distance of NMF estimate from target matrix is 33283.360. Random VCol initialization algorithm is used. 
       
       
    .. figure:: /images/orl_faces_200_iters_small_LSNMF.png
       :scale: 70 %
       :alt: Basis images of LSNMF obtained after 200 iterations on reduced face images.  
       :align: center

       Basis images of LSNMF obtained after 200 iterations on reduced face images. The bases trained by LSNMF are additive. The
       Euclidean distance of LSNMF estimate from target matrix is 29631.784 and projected gradient norm, which is used as 
       objective function in LSNMF is 7.9. Random VCol initialization algorithm is used. In LSNMF there is parameter beta, 
       we set is to 0.1. Beta is the rate of reducing the step size to satisfy the sufficient decrease condition. Smaller
       beta reduces the step size aggressively but may result in step size that is too small and the cost per iteration is thus
       higher. 
    
       
    .. figure:: /images/orl_faces_5_iters_small_PSMF_prior5.png
       :scale: 70 %
       :alt: Basis images of PSMF obtained after 5 iterations on reduced face images and with set prior parameter to 5.  
       :align: center

       Basis images of PSMF obtained after 5 iterations on reduced face images and with set prior parameter to 5. The
       bases trained from PSMF are both additive and spatially localized for representing faces. By setting prior to 5, in PSMF 
       the basis matrix is found under structural sparseness constraint that each row contains at most 5 non zero entries. This
       means, each row vector of target data matrix is explained by linear combination of at most 5 factors. Because we passed 
       prior as scalar and not list, uniform prior is taken, reflecting no prior knowledge on the distribution.  
       
       
    To run the example simply type::
        
        python orl_images.py
        
    or call the module's function::
    
        import nimfa.examples
        nimfa.examples.orl_images.run()
        
    .. note:: This example uses ``matplotlib`` library for producing visual interpretation of basis vectors. It uses PIL 
              library for displaying face images. 
"""

from os.path import dirname, abspath
from os.path import join
from warnings import warn

import numpy as np

import nimfa

try:
    from matplotlib.pyplot import savefig, imshow, set_cmap
except ImportError as exc:
    warn("Matplotlib must be installed to run ORL images example.")

try:
    from PIL.Image import open, fromarray, new
    from PIL.ImageOps import expand
except ImportError as exc:
    warn("PIL must be installed to run ORL images example.")


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
    lsnmf = nimfa.Lsnmf(V, seed="random_vcol", rank=25, max_iter=50, sub_iter=10,
                        inner_sub_iter=10, beta=0.1, min_residuals=1e-8)
    print("Algorithm: %s\nInitialization: %s\nRank: %d" % (lsnmf, lsnmf.seed, lsnmf.rank))
    fit = lsnmf()
    print("""Stats:
            - iterations: %d
            - final projected gradients norm: %5.3f
            - Euclidean distance: %5.3f""" % (fit.fit.n_iter, fit.distance(),
                                              fit.distance(metric='euclidean')))
    return fit.basis(), fit.coef()


def read():
    """
    Read face image data from the ORL database. The matrix's shape is 2576 (pixels) x 400 (faces). 
    
    Step through each subject and each image. Reduce the size of the images by a factor of 0.5. 
    
    Return the ORL faces data matrix. 
    """
    print("Reading ORL faces database")
    dir = join(dirname(dirname(abspath(__file__))), 'datasets', 'ORL_faces', 's')
    V = np.matrix(np.zeros((46 * 56, 400)))
    for subject in range(40):
        for image in range(10):
            im = open(join(dir + str(subject + 1), str(image + 1) + ".pgm"))
            # reduce the size of the image
            im = im.resize((46, 56))
            V[:, image * subject + image] = np.mat(np.asarray(im).flatten()).T
    return V


def preprocess(V):
    """
    Preprocess ORL faces data matrix as Stan Li, et. al.
    
    Return normalized and preprocessed data matrix. 
    
    :param V: The ORL faces data matrix. 
    :type V: `numpy.matrix`
    """
    print("Data preprocessing")
    min_val = V.min(axis=0)
    V = V - np.mat(np.ones((V.shape[0], 1))) * min_val
    max_val = V.max(axis=0) + 1e-4
    V = (255. * V) / (np.mat(np.ones((V.shape[0], 1))) * max_val) / 100.
    return V


def plot(W):
    """
    Plot basis vectors.
    
    :param W: Basis matrix of the fitted factorization model.
    :type W: `numpy.matrix`
    """
    set_cmap('gray')
    blank = new("L", (225 + 6, 280 + 6))
    for i in range(5):
        for j in range(5):
            basis = np.array(W[:, 5 * i + j])[:, 0].reshape((56, 46))
            basis = basis / np.max(basis) * 255
            basis = 255 - basis
            ima = fromarray(basis)
            expand(ima, border=1, fill='black')
            blank.paste(ima.copy(), (j * 46 + j, i * 56 + i))
    imshow(blank)
    savefig("orl_faces.png")


if __name__ == "__main__":
    """Run the ORL faces example."""
    run()
