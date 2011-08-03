import scipy.sparse as sp

class Random(object):
    """
    Random is the simplest MF initialization method.
    """
    
    def __init__(self):
        self.name = "random"
       
    def initialize(self, V, rank, **options):
        """
        Return initialized basis and mixture matrix. 
        
        :param V: Target matrix, the matrix for MF method to estimate.
        :type V: One of the :class:`scipy.sparse` sparse matrices types or :class:`numpy.ndarray` or or :class:`numpy.matrix`
        :param rank: Factorization rank. 
        :type rank: `int`
        :param options: Specify algorithm or model specific options (e.g. initialization of extra matrix factor, seeding parameters).
                        Options 'Sn', n = 1, 2, 3, ... specify additional matrix factors which need to be initialized.
                        The value of each option is a tuple, denoting matrix shape. Matrix factors are returned in the same
                        order as their descriptions in input. 
        :type options: `dict`
        """
        self.V = V
        self.rank = rank
        self.W = sp.rand(self.V.shape[0], self.rank, density = 0.01, format = 'csr', dtype = 'd')
        self.H = sp.rand(self.rank, self.V.shape[1], density = 0.01, format = 'csr', dtype = 'd')
        mfs = [self.W, self.H]
        for sn in options:
            if sn[0] is 'S' and sn[1:].isdigit():
                mfs.append(sp.rand(sn[0], sn[1], density = 0.01, format = 'csr', dtype = 'd'))
        return mfs
    
    def __repr__(self):
        return "random.Random()"
    
    def __str__(self):
        return self.name