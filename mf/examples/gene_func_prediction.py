
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
           FunCat. In example is used data set that originates from S. cerevisiae and has annotations from the MIPS Functional
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
    """Run the NMF - Divergence on the S. cerevisiae sequence data set."""
    data, test, idx2attr, idx2class = read()
    data = preprocess(data)
    data = factorize(data)

def read():
    """
    Read S. cerevisiae FunCat annotated sequence data set.
    
    Return attributes' values and class information of the test data set and joined train and validation data set. Additional mapping functions 
    are returned mapping attributes' names and classes' names to indices. 
    """
    print "Reading S. cerevisiae FunCat annotated sequence data set ..."
    dir = dirname(dirname(abspath(__file__))) + sep + 'datasets' + sep + 'S_cerevisiae_FC' + sep + 'seq_yeast_FUN' + sep
    train_data = dir + 'seq_yeast_FUN.train.arff'
    valid_data = dir + 'seq_yeast_FUN.valid.arff'
    test_data = dir + 'seq_yeast_FUN.test.arff'
    print " Reading S. cerevisiae FunCat annotated sequence TRAIN set ..."
    train, idx2attr, idx2class = transform_data(train_data, include_meta = True)
    print " Reading S. cerevisiae FunCat annotated sequence VALIDATION set ..."
    valid = transform_data(valid_data)
    print " Reading S. cerevisiae FunCat annotated sequence TEST set ..."
    test = transform_data(test_data)
    print " ... Finished."
    print " Joining S. cerevisiae FunCat annotated sequence TEST and VALIDATION set ..."
    tv_data = _join(train, valid)
    print " ... Finished."    
    return tv_data, test, idx2attr, idx2class

def transform_data(path, include_meta = False):
    """
    Read data in the ARFF format and transform it to suitable matrix for factorization process.
    
    Return attributes' values and class information. If :param:`include_meta` is specified additional mapping functions are provided with 
    attributes' names and classes' names.  
    
    :param path: Path of directory with sequence data set.
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
    feature = 0
    used_idx = set()
    section = 'h'

    for line in open(path):
        if section == 'h': 
            tokens = line.strip().split()
            line_type = tokens[0] if tokens else None
            if line_type == "@ATTRIBUTE":
                if tokens[2] in ["numeric"]:
                    attr2idx[tokens[1]] = idx_attr
                    idx_attr += 1
                    used_idx.add(idx)
                if tokens[1] in ["class"] and tokens[2] in ["hierarchical", "classes"]:
                    class2idx = _reverse(dict(list(enumerate((tokens[3] if tokens[3] != '%' else tokens[5]).split(",")))))
                    idx_class = idx
                idx += 1
            if line_type == "@DATA":
                section = 'd'
                idxs = set(xrange(idx)).intersection(used_idx)
                attr_data = np.mat(np.zeros((1e4, len(attr2idx))))
                class_data = np.mat(np.zeros((1e4, len(class2idx))))
        elif section == 'd':
            d, _, comment = line.strip().partition("%")
            values = d.split(",")
            # update class information for current feature
            class_var = map(str.strip, values[idx_class].split("@"))
            for cl in class_var:
                class_data[feature, class2idx[cl]] = 1.0
            # update attribute values information for current feature 
            i = 0 
            for idx in idxs:
                attr_data[feature, i] = float(values[idx] if values[idx] != '?' else 0.)
                i += 1
            feature += 1
    return ({'feat': feature, 'attr': attr_data, 'class': class_data}, _reverse(attr2idx), _reverse(class2idx)) if include_meta else {'feat': feature, 'attr': attr_data, 'class': class_data}

def _join(train, valid):
    """
    Join test and validation data of the S. cerevisiae FunCat annotated sequence data set. 
    
    Return joined test and validation attributes' values and class information.
     
    :param train: Attributes' values and class information of the train data set. 
    :type train: `numpy.matrix`
    :param valid: Attributes' values and class information of the validation data set.
    :type valid: `numpy.matrix`
    """
    n_train =  train['feat']
    n_valid =  valid['feat']
    return {'attr': np.vstack((train['attr'][:n_train, :], valid['attr'][:n_valid, :])),
            'class': np.vstack((train['class'][:n_train, :], valid['class'][:n_valid, :]))}

def _reverse(object2idx):
    """
    Reverse mapping function.
    
    Return reversed mapping.
    
    :param object2idx: Mapping of objects to indices or vice verse.
    :type object2idx: `dict`
    """
    return dict(zip(object2idx.values(), object2idx.keys()))

def factorize(data):
    """
    Perform factorization on S. cerevisiae FunCat annotated sequence data set.
    
    Return factorized data. 
    
    :param data: Transformed data set containing attributes' values, class information and possibly additional meta information.  
    :type data: `tuple`
    """
    print "Performing NMF - Divergence factorization ..." 
    print "... Finished."

def preprocess(data):
    """
    Preprocess S.cerevisiae FunCat annotated sequence data set. Preprocessing step includes building matrix exposing
    hierarchy constraints of FunCat annotations.
    
    Return preprocessed data. 
    
    :param data: Transformed data set containing attributes' values, class information and possibly additional meta information.  
    :type data: `tuple`
    """
    print "Preprocessing data matrix ..."
    print "... Finished."

if __name__ == "__main__": 
    """Run the gene function prediction example."""
    run()



    