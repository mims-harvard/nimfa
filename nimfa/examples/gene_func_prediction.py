
"""
    ############################################################
    Gene Function Prediction (``examples.gene_func_prediction``)
    ############################################################
    
    As a background reading before this example, we recommend user to read [Schietgat2010]_ and [Schachtner2008]_ where
    the authors study the use of decision tree based models for predicting the multiple gene functions and unsupervised 
    matrix factorization techniques to extract marker genes from gene expression profiles for classification into
    diagnostic categories, respectively. 
        
    This example from functional genomics deals with predicting gene functions. Two main characteristics of gene function 
    prediction task are:
    
        #. single gene can have multiple functions, 
        #. the functions are organized in a hierarchy, in particular in a hierarchy structered as a rooted tree -- MIPS
           Functional Catalogue. A gene related to some function is automatically related to all its ancestor 
           functions. Data set used in this example originates from S. cerevisiae and has annotations from the MIPS 
           Functional Catalogue. 
    
    The latter problem setting describes hierarchical multi-label classification (HMC).
    
    .. note:: The S. cerevisiae FunCat annotated data set used in this example is not included in the `datasets`. If you 
              wish to perform the gene function prediction experiments, start by downloading the data set. In particular
              D1 (FC) seq data set must be available for the example to run.  Download links are listed in the 
              ``datasets``. To run the example, uncompress the data and put it into corresponding data directory, namely 
              the extracted data set must exist in the ``S_cerevisiae_FC`` directory under ``datasets``. Once you have 
              the data installed, you are ready to start running the experiments.  
    
    Here is the outline of this gene function prediction task. 
    
        #. Reading S. cerevisiae sequence data, i. e. train, validation and test set. Reading meta data,  
           attributes' labels and class labels. Weights are used to distinguish direct and indirect class 
           memberships of genes in gene function classes according to FunCat annotations. 
        #. Preprocessing, i. e. normalizing data matrix of test data and data matrix of joined train and validation
           data. 
        #. Factorization of train data matrix. We used SNMF/L factorization algorithm for train data. 
        #. Factorization of test data matrix. We used SNMF/L factorization algorithm for train data.
        #. Application of rules for class assignments. Three rules can be used, average correlation and maximal 
           correlation, as in [Schachtner2008]_ and threshold maximal correlation. All class assignments rules
           are generalized to meet the hierarchy constraint imposed by the rooted tree structure of MIPS Functional 
           Catalogue. 
        #. Precision-recall (PR) evaluation measures. 
    
    To run the example simply type::
        
        python gene_func_prediction.py
        
    or call the module's function::
    
        import nimfa.examples
        nimfa.examples.gene_func_prediction.run()
        
    .. note:: This example uses ``matplotlib`` library for producing visual interpretation.
"""

from os.path import dirname, abspath
from os.path import join
from warnings import warn

import numpy as np

import nimfa

try:
    import matplotlib.pylab as plb
except ImportError as exc:
    warn("Matplotlib must be installed to run Gene Function prediction example.")


def run():
    """
    Run the gene function prediction example on the S. cerevisiae sequence data set (D1 FC seq).
    
    The methodology is as follows:
        #. Reading S. cerevisiae sequence data, i. e. train, validation and test set. Reading meta data,  
           attributes' labels and class labels.
        #. Preprocessing, i. e. normalizing data matrix of test data and data matrix of joined train and validation
           data. 
        #. Factorization of train data matrix. We used SNMF/L factorization algorithm for train data. 
        #. Factorization of test data matrix. We used SNMF/L factorization algorithm for train data.
        #. Application of rules for class assignments. Three rules can be used, average correlation and maximal 
           correlation, as in [Schachtner2008]_ and threshold maximal correlation. All class assignments rules
           are generalized to meet the hierarchy constraint imposed by the rooted tree structure of MIPS Functional 
           Catalogue. 
        #. PR evaluation measures. 
    """
    tv_data, test_data, idx2attr, idx2class = read()
    tv_data = preprocess(tv_data)
    test_data = preprocess(test_data)
    tv_data = factorize(tv_data)
    test_data = factorize(test_data)
    corrs = compute_correlations(tv_data, test_data)
    for method in 0.5 * np.random.random_sample(50) + 1.:
        func2gene = assign_labels(corrs, tv_data, idx2class, method=method)
        plot(func2gene, test_data, idx2class)


def read():
    """
    Read S. cerevisiae FunCat annotated sequence data set (D1 FC seq).
    
    Return attributes' values and class information of the test data set and joined train and validation data set. Additional mapping functions 
    are returned mapping attributes' names and classes' names to indices. 
    """
    print(" Reading S. cerevisiae FunCat annotated sequence data set (D1 FC seq)")
    dir = join(dirname(dirname(abspath(__file__))), 'datasets', 'S_cerevisiae_FC',  'seq_yeast_FUN')
    train_data = join(dir, 'seq_yeast_FUN.train.arff')
    valid_data = join(dir, 'seq_yeast_FUN.valid.arff')
    test_data = join(dir, 'seq_yeast_FUN.test.arff')
    print(" Reading S. cerevisiae FunCat annotated sequence (D1 FC seq) TRAIN set")
    train, idx2attr, idx2class = transform_data(
        train_data, include_meta=True)
    print(" Reading S. cerevisiae FunCat annotated sequence (D1 FC seq) VALIDATION set")
    valid = transform_data(valid_data)
    print(" Reading S. cerevisiae FunCat annotated sequence (D1 FC seq) TEST set")
    test = transform_data(test_data)
    print(" Joining S. cerevisiae FunCat annotated sequence (D1 FC seq) TEST and VALIDATION set")
    tv_data = _join(train, valid)
    return tv_data, test, idx2attr, idx2class


def transform_data(path, include_meta=False):
    """
    Read data in the ARFF format and transform it to suitable matrix for factorization process. For each feature update direct and indirect 
    class information exploiting properties of Functional Catalogue hierarchy. 
    
    Return attributes' values and class information. If :param:`include_meta` is specified additional mapping functions are provided with 
    mapping from indices to attributes' names and indices to classes' names.  
    
    :param path: Path of directory with sequence data set (D1 FC seq).
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
                    class2idx = _reverse(
                        dict(list(enumerate((tokens[3] if tokens[3] != '%' else tokens[5]).split(",")))))
                    idx_class = idx
                idx += 1
            if line_type == "@DATA":
                section = 'd'
                idxs = set(range(idx)).intersection(used_idx)
                attr_data = np.mat(np.zeros((1e4, len(attr2idx))))
                class_data = np.mat(np.zeros((1e4, len(class2idx))))
        elif section == 'd':
            d, _, comment = line.strip().partition("%")
            values = d.split(",")
            # update class information for current feature
            class_var = list(map(str.strip, values[idx_class].split("@")))
            for cl in class_var:
                # update direct class information
                class_data[feature, class2idx[cl]] += 10.
                # update indirect class information through FunCat hierarchy
                cl_a = cl.split("/")
                cl = "/".join(cl_a[:3] + ['0'])
                if cl in class2idx:
                    class_data[feature, class2idx[cl]] += 3.
                cl = "/".join(cl_a[:2] + ['0', '0'])
                if cl in class2idx:
                    class_data[feature, class2idx[cl]] += 2.
                cl = "/".join(cl_a[:1] + ['0', '0', '0'])
                if cl in class2idx:
                    class_data[feature, class2idx[cl]] += 1.
            # update attribute values information for current feature
            i = 0
            for idx in idxs:
                attr_data[feature, i] = abs(
                    float(values[idx] if values[idx] != '?' else 0.))
                i += 1
            feature += 1
    if include_meta:
        data = ({'feat': feature, 'attr': attr_data[:feature, :], 'class': class_data[:feature, :]},
                _reverse(attr2idx), _reverse(class2idx))
    else:
        data = {'feat': feature, 'attr': attr_data[:feature, :], 'class': class_data[:feature, :]}
    return data


def _join(train, valid):
    """
    Join test and validation data of the S. cerevisiae FunCat annotated sequence data set (D1 FC seq). 
    
    Return joined test and validation attributes' values and class information.
     
    :param train: Attributes' values and class information of the train data set. 
    :type train: `numpy.matrix`
    :param valid: Attributes' values and class information of the validation data set.
    :type valid: `numpy.matrix`
    """
    n_train = train['feat']
    n_valid = valid['feat']
    return {'feat': n_train + n_valid,
            'attr': np.vstack((train['attr'][:n_train, :], valid['attr'][:n_valid, :])),
            'class': np.vstack((train['class'][:n_train, :], valid['class'][:n_valid, :]))}


def _reverse(object2idx):
    """
    Reverse 1-to-1 mapping function.
    
    Return reversed mapping.
    
    :param object2idx: Mapping of objects to indices or vice verse.
    :type object2idx: `dict`
    :rtype: `dict`
    """
    return dict(list(zip(list(object2idx.values()), list(object2idx.keys()))))


def preprocess(data):
    """
    Preprocess S.cerevisiae FunCat annotated sequence data set (D1 FC seq). Preprocessing step includes data 
    normalization.
    
    Return preprocessed data. 
    
    :param data: Transformed data set containing attributes' values, class information and possibly additional meta information.  
    :type data: `tuple`
    """
    print("Data preprocessing")
    data['attr'] = (data['attr'] - data['attr'].min() + np.finfo(
        data['attr'].dtype).eps) / (data['attr'].max() - data['attr'].min())
    return data


def factorize(data):
    """
    Perform factorization on S. cerevisiae FunCat annotated sequence data set (D1 FC seq).
    
    Return factorized data, this is matrix factors as result of factorization (basis and mixture matrix). 
    
    :param data: Transformed data set containing attributes' values, class information and possibly additional meta information.  
    :type data: `tuple`
    """
    V = data['attr']
    snmf = nimfa.Snmf(V, seed="random_vcol", rank=40, max_iter=5, version="l", eta=1., beta=1e-4,
                     i_conv=10, w_min_change=0)
    print("Algorithm: %s\nInitialization: %s\nRank: %d" % (snmf, snmf.seed, snmf.rank))
    fit = snmf()
    sparse_w, sparse_h = fit.fit.sparseness()
    print("""Stats:
            - iterations: %d
            - KL Divergence: %5.3f
            - Euclidean distance: %5.3f
            - Sparseness basis: %5.3f, mixture: %5.3f""" % (fit.fit.n_iter, fit.distance(), fit.distance(metric='euclidean'), sparse_w, sparse_h))
    data['W'] = fit.basis()
    data['H'] = fit.coef()
    return data


def compute_correlations(train, test):
    """
    Estimate correlation coefficients between profiles of train basis matrix and profiles of test basis matrix. 
    
    Return the estimated correlation coefficients of the features (variables).  
    
    :param train: Factorization matrix factors of train data set. 
    :type train: `dict`
    :param test: Factorization matrix factors of test data set. 
    :type test: `dict`
    :rtype: `numpy.matrix`
    """
    print("Estimating correlation coefficients")
    corrs = np.corrcoef(train['W'], test['W'])
    # alternative, it is time consuming - can be used for partial evaluation
    """corrs = {}
    for i in xrange(test['W'].shape[0]):
        corrs.setdefault(i, np.mat(np.zeros((train['W'].shape[0], 1))))
        for j in xrange(train['W'].shape[0]):
            corrs[i][j, 0] = _corr(test['W'][i, :], train['W'][j, :])"""
    return np.mat(corrs)


def _corr(x, y):
    """
    Compute Pearson's correlation coefficient of x and y. Numerically stable algebraically equivalent equation for 
    coefficient computation is used. 
    
    Return correlation coefficient between x and y which is by definition in [-1, 1].
    
    :param x: Random variable.
    :type x: `numpy.matrix`
    :param y: Random variable.
    :type y: `numpy.matrix`
    :rtype: `float`
    """
    xc = (x - x.mean()) / x.std(ddof=1)
    yc = (y - y.mean()) / y.std(ddof=1)
    return 1. / (x.size - 1) * np.multiply(xc, yc).sum()


def assign_labels(corrs, train, idx2class, method=0.):
    """
    Apply rules for class assignments. In [Schachtner2008]_ two rules are proposed, average correlation and maximal 
    correlation. Here, both the rules are implemented and can be specified through :param:`method``parameter. In addition to 
    these the threshold maximal correlation rule is possible as well. Class assignments rules are generalized to 
    multi-label classification incorporating hierarchy constraints. 
    
    User can specify the usage of one of the following rules:
        #. average correlation,
        #. maximal correlation,
        #. threshold maximal correlation.
    
    Though any method based on similarity measures can be used, we estimate correlation coefficients. Let w be the
    gene profile of test basis matrix for which we want to predict gene functions. For each class C a separate 
    index set A of indices is created, where A encompasses all indices m, for which m-th profile of train basis 
    matrix has label C. Index set B contains all remaining indices. Now, the average correlation coefficient between w
    and elements of A is computed, similarly average correlation coefficient between w and elements of B. Finally, 
    w is assigned label C if the former correlation over the respective index set is greater than the 
    latter correlation.
    
    .. note:: Described rule assigns the class label according to an average correlation of test vector with all
              vectors belonging to one or the other index set. Minor modification of this rule is to assign the class
              label according to the maximal correlation occurring between the test vector and the members of each
              index set. 
             
    .. note:: As noted before the main problem of this example is the HMC (hierarchical multi-label classification) 
              setting. Therefore we generalized the concepts from articles describing the use of factorization
              for binary classification problems to multi-label classification. Additionally, we use the weights
              for class memberships to incorporate hierarchical structure of MIPS MIPS Functional
              Catalogue.
    
    Return mapping of gene functions to genes.  
    
    :param corrs: Estimated correlation coefficients between profiles of train basis matrix and profiles of test 
                  basis matrix. 
    :type corrs: `dict`
    :param train: Class information of train data set. 
    :type train: `dict`
    :param idx2class: Mapping between classes' indices and classes' labels. 
    :type idx2class: `dict`
    :param method: Type of rule for class assignments. Possible are average correlation, maximal correlation by 
                   specifying ``average`` or ``maximal`` respectively. In addition threshold maximal correlation is
                   supported. If threshold rule is desired, threshold is specified instead. By default 
                   threshold rule is applied. 
    :type method: `float` or `str`
    :rtype: `dict`
    """
    print("Assigning class labels - gene functions to genes")
    func2gene = {}
    n_train = train['feat']
    n_cl = len(idx2class)
    for cl_idx in range(n_cl):
        func2gene.setdefault(cl_idx, [])
    key = 0
    for test_idx in range(n_train, corrs.shape[0]):
        if method == "average":
            # weighted summation of correlations over respective index sets
            avg_corr_A = np.sum(
                np.multiply(np.tile(corrs[:n_train, test_idx], (1, n_cl)), train['class']), 0)
            avg_corr_B = np.sum(
                np.multiply(np.tile(corrs[:n_train, test_idx], (1, n_cl)), train['class'] != 0), 0)
            avg_corr_A = avg_corr_A / (np.sum(train['class'] != 0, 0) + 1)
            avg_corr_B = avg_corr_B / (np.sum(train['class'] == 0, 0) + 1)
            for cl_idx in range(n_cl):
                if (avg_corr_A[0, cl_idx] > avg_corr_B[0, cl_idx]):
                    func2gene[cl_idx].append(key)
        elif method == "maximal":
            max_corr_A = np.amax(
                np.multiply(np.tile(corrs[:n_train, test_idx], (1, n_cl)), train['class']), 0)
            max_corr_B = np.amax(
                np.multiply(np.tile(corrs[:n_train, test_idx], (1, n_cl)), train['class'] != 0), 0)
            for cl_idx in range(n_cl):
                if (max_corr_A[0, cl_idx] > max_corr_B[0, cl_idx]):
                    func2gene[cl_idx].append(key)
        elif isinstance(method, float):
            max_corr = np.amax(
                np.multiply(np.tile(corrs[:n_train, test_idx], (1, n_cl)), train['class']), 0)
            for cl_idx in range(n_cl):
                if (max_corr[0, cl_idx] >= method):
                    func2gene[cl_idx].append(key)
        else:
            raise ValueError("Unrecognized class assignment rule.")
        key += 1
        if key % 100 == 0:
            print(" %d/%d" % (key, corrs.shape[0] - n_train))
    return func2gene


def plot(func2gene, test, idx2class):
    """
    Report the performance with the precision-recall (PR) based evaluation measures. 
    
    Beside PR also ROC based evaluations have been used before to evaluate gene function prediction approaches. PR
    based better suits the characteristics of the common HMC task, in which many classes are infrequent with a small
    number of genes having particular function. That is for most classes the number of negative instances exceeds
    the number of positive instances. Therefore it is sometimes preferred to recognize the positive instances instead
    of correctly predicting the negative ones (i. e. gene does not have a particular function). That means that ROC
    curve might be less suited for the task as they reward a learner if it correctly predicts negative instances. 
    
    Return PR evaluations measures
    
    :param labels: Mapping of genes to their predicted gene functions. 
    :type labels: `dict`
    :param test: Class information of test data set. 
    :type test: `dict`
    :param idx2class: Mapping between classes' indices and classes' labels. 
    :type idx2class: `dict`
    :rtype: `tuple`
    """
    print("Computing PR evaluations measures")

    def tp(g_function):
        # number of true positives for g_function (correctly predicted positive
        # instances)
        return (test['class'][func2gene[g_function], g_function] != 0).sum()

    def fp(g_function):
        # number of false positives for g_function (positive predictions that
        # are incorrect)
        return (test['class'][func2gene[g_function], g_function] == 0).sum()

    def fn(g_function):
        # number of false negatives for g_function (positive instances that are
        # incorrectly predicted negative)
        n_pred = list(
            set(range(len(idx2class))).difference(func2gene[g_function]))
        return (test['class'][n_pred, g_function] != 0).sum()
    tp_sum = 0.
    fp_sum = 0.
    fn_sum = 0.
    for g_function in idx2class:
        tp_sum += tp(g_function)
        fp_sum += fp(g_function)
        fn_sum += fn(g_function)
    avg_precision = tp_sum / (tp_sum + fp_sum)
    avg_recall = tp_sum / (tp_sum + fn_sum)
    print("Average precision over all gene functions: %5.3f" % avg_precision)
    print("Average recall over all gene functions: %5.3f" % avg_recall)
    return avg_precision, avg_recall


if __name__ == "__main__":
    """Run the gene function prediction example."""
    run()
