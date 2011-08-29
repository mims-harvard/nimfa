
"""
    ####################################
    Orl_images (``examples.orl_images``)
    ####################################
    
    
    .. note:: ORL face images database used in this example is included in the `datasets` and need not to be
          downloaded. However, download links are listed in the ``datasets``. To run the example, the ORL face images
          must be find in the `ORL_faces` folder under `datasets`. 
"""

import mf
import numpy as np
from matplotlib.pyplot import savefig, imshow, set_cmap
from os.path import dirname, abspath

def run():
    """Run LSNMF on ORL faces data set."""
    read()
    
    
def read():
    """Read face image date from ORL database. Step through each subject and each image."""
    dir = dirname(dirname(abspath(__file__)))
    for subject in xrange(40):
        for image in xrange(10):
            data = read_pgm(dir + '/datasets/ORL_faces/s' + str(subject +1) + "/" + str(image + 1) + ".pgm")
            print data.shape
            exit()
    

def reduce(data, factor = 0.5):
    """
    Reduce the size of the image data by a given factor.
    
    :param data: Image data which size will be reduced. 
    :type data: `numpy.ndarray`
    :param factor: Resize factor. Default is 0.5. 
    :type factor: `float`
    """    
    pass

def read_pgm(path):
    """
    Open a raw PGM file and read the data. Each PGM image consists of the following: 
        #. a magic number for identifying the file type (i. e. P5 in the normal format or P2 in the plain format);
        #. width and height, formatted ad ASCII;
        #. the maximum gray value formatted in ASCII;
        #. a raster of height rows, in order from top to bottom and each row consists of width gray values, in order
           from left to right.
    
    Return numpy array containing image data.
    
    :param path: Path to image in raw PGM format.
    :type path: `str`
    """
    f = open(path, 'rb')
    magic = f.readline().rstrip()
    w, h = map(int, f.readline().rstrip().split())
    max_val = int(f.readline().rstrip())
    # check consistency with the ORL database
    if magic != 'P5' or w != 92 or h != 112 or max_val != 255:
        return None
    data = np.fromfile(f, dtype = np.uint8)
    return np.reshape(data, (92, 112))

if __name__ == "__main__":
    """Run the ORL faces example."""
    run()