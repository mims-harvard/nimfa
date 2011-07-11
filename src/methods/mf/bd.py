
class Bd(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        self.aname = "bd"
        self.amodels = ["nmf_std"]
        self.aseeds = ["nndsvd"]
        
    def factorize(self, model):
        self.__dict__.update(model.__dict__)
        