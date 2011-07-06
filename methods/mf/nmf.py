
class Nmf(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.name = "nmf"
        self.amodels = ["nmf_std"]
        self.aseeds = ["nnsvd"]
        
    def factorize(self, model):
        self.__dict__.update(model.__dict__)
    
        