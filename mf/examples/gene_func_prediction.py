
"""
    ########################################################
    Gene_func_prediction (``examples.gene_func_prediction``)
    ########################################################
    
    .. note:: This example is in progress.
    
    As a background reading before this example, we suggest reading [Schietgat2010]_ and [Schachtner2008]_
        
    This example from functional genomics deals with gene function prediction. Two main characteristics of function 
    prediction task are:
    
        #. single gene can have multiple functions, 
        #. the functions are organized in a hierarchy, in particular in a hierarchy structered as a rooted tree -- MIPS's
           FunCat. In example is used dataset that originates from S. cerevisiae and has annotations from the MIPS Functional
           Catalogue. A gene related to some function is automatically related to all its ancestor functions.
    
    These characteristics describe hierarchical multi-label classification setting. 
    
    Here is the outline of this gene function prediction task. 
    
        #. Dataset Preprocessing.
        #. Gene selection
        #. Feature generation. 
        #. Feature selection
        #. Classification of the mixture matrix and comply with the hierarchy constraint. 
    
    To run the example simply type::
        
        python gene_func_prediction.py
        
    or call the module's function::
    
        import mf.examples
        mf.examples.gene_func_prediction.run()
        
    .. note:: This example uses matplotlib library for producing visual interpretation.
"""

import mf
import numpy as np
import scipy.sparse as sp
from os.path import dirname, abspath, sep

try:
    import matplotlib.pylab as plb
except ImportError, exc:
    raise SystemExit("Matplotlib must be installed to run this example.")
    

def run():
    """Run the NMF - Divergence on the S. cerevisiae sequence dataset."""
    pass

def read():
    """Read S. cerevisiae FunCat annotated sequence dataset."""
    print "Reading S. cerevisiae FunCat annotated sequence dataset ..."
    dir = dirname(dirname(abspath(__file__))) + sep + 'datasets' + sep + 'S_cerevisiae_FC' + sep
    train_data = dir + 'seq_yeast_FUN.train.arff'
    valid_data = dir + 'seq_yeast_FUN.valid.arff'
    test_data = dir + 'seq_yeast_FUN.test.arff'
    train_attr_data, train_class_data, idx2attr, idx2class = transform_data(train_data, include_meta = True)
    valid_attr_data, valid_class_data = transform_data(valid_data)
    test_attr_data, test_class_data = transform_data(test_data)
    print "... Finished."
    return train_attr_data, train_class_data, idx2attr, idx2class, valid_attr_data, valid_class_data, test_attr_data, test_class_data

def transform_data(path, include_meta = False):
    """
    Read data in the ARFF format and transform it to suitable matrix for factorization process.
    
    Return attributes values and class information. If :param:`include_meta` is specified additional mapping functions are provided with 
    attributes' names and classes' names.  
    
    :param path: Path of directory with sequence dataset.
    :type path: `str`
    :param include_meta: Specify if the header of the ARFF file should be skipped. The header of the ARFF file 
                               contains the name of the relation, a list of the attributes and their types. Default
                               value is False.  
    :type include_meta: `bool`
    """
    class2idx = {}
    attr2idx = {}
    
    idx_attr = 0
    idx_class = 0
    idx = 0
    gene = 0
    section = HEADER

    for line in open(path):
        if section == HEADER: 
            tokens = line.strip().split()
            line_type = tokens[0] if tokens else None
            if line_type == "@ATTRIBUTE":
                if tokens[2] in ["numeric"]:
                    attr2idx[tokens[1]] = idx_attr
                    idx_attr += 1
                if tokens[1] in ["class"] and tokens[2] in ["hierarchical", "classes"]:
                    class2idx = dict(list(enumerate(tokens[3].split(","))))
                    idx_class = idx
            idx += 1
            if line_type == "@DATA":
                section = DATA
                attr_data = np.mat(np.zeros((1e4, len(attr2idx))))
                class_data = np.mat(np.zeros((1e-4, len(class2idx))))
        elif section == DATA:
            d, comment = line.strip().partition("%")
            values = d.split(",")
            # update class information for current gene
            class_var = map(str.strip, values[idx_class].split("@"))
            for cl in class_var:
                class_data[gene, class2idx[cl]] = 1.0
            # update attribute values information for current gene 
            idxs = set(xrange(len(values))).intersection(attr2idx.values())
            i = 0 
            if idx in idxs:
                data[gene, i] = double(values[idx])
                i += 1
            gene += 1
    return attr_data, class_data if not include_attributes else attr_data, class_data, _reverse(attr2idx), _reverse(class2idx)

def _reverse(object2idx):
    """
    Reverse mapping function (objects --> indices).
    
    Return reversed mapping.
    
    :param object2idx: Mapping of objects to indices.
    :type object2idx: `dict`
    """
    return dict(zip(object2idx.values(), object2idx.keys()))

def factorize():
    """Perform factorization on S. cerevisiae FunCat annotated sequence dataset."""
    pass

def preprocess():
    """
    Preprocess S.cerevisiae FunCat annotated sequence dataset. Preprocessing step includes building matrix exposing
    hierarchy constraints of FunCat annotations.
    """
    pass

if __name__ == "__main__": 
    """Run the gene function prediction example."""
    run()



    