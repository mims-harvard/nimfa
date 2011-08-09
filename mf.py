
"""
    This module implements the main interface to launch matrix factorization algorithms. 
    MF algorithms can be combined with implemented seeding methods.
    
    Returned object can be directly passed to visualization or comparison utilities or as initialization 
    to another factorization method.
    
    #. [mandatory] Choose the MF model by specifying the algorithm to perform MF on target matrix.
    #. Choose the number of runs of the MF algorithm. Useful for achieving stability when using random
       seeding method.  
    #. Pass a callback function which is called after each run when performing multiple runs of the algorithm.
       Useful for saving summary measures or processing the result of each NMF fit before it gets discarded. The
       callback function is called after each run.  
    #. [mandatory] Choose the factorization rank to achieve.
    #. [mandatory] Choose the seeding method to compute the starting point passed to te algorithm. 
    #. [mandatory] Provide the target object to estimate. 
"""
import methods
import utils.utils as utils

l_factorization = methods.list_mf_methods()
l_seed = methods.list_seeding_methods()

def mf(target, seed = None, W = None, H = None,  
       rank = 30, method = "nmf",
       max_iter = 30, min_residuals = None, test_conv = None,
       n_run = 1, callback = None, initialize_only = False, **options):
    """
    Run the specified MF algorithm.
    
    Return fitted factorization model storing MF results. If :param:`initialize_only` is set, only initialized model is returned.
    
    :param target: The target matrix to estimate.
    :type target: One of the :class:`scipy.sparse` sparse matrices types or :class:`numpy.ndarray` or :class:`numpy.matrix` 
    :param seed: Specify method to seed the computation of a factorization. If specified :param:`W` and :param:`H` seeding 
                 must be None. If neither seeding method or initial fixed factorization is specified, random initialization is used
    :type seed: `str` naming the method or :class:`methods.seeding.nndsvd.Nndsvd` or None
    :param W: Specify initial factorization of basis matrix W. Default is None. When specified, :param:`seed` must be None.
    :type W: :class:`scipy.sparse` or :class:`numpy.ndarray` or :class:`numpy.matrix` or None
    :param H: Specify initial factorization of mixture matrix H. Default is None. When specified, :param:`seed` must be None.
    :type H: :class:`scipy.sparse` or :class:`numpy.ndarray` or :class:`numpy.matrix` or None
    :param rank: The factorization rank to achieve. Default is 30.
    :type rank: `int`
    :param method: The algorithm to use to perform MF on target matrix. Default is :class:`methods.mf.nmf`
    :type method: `str` naming the algorithm or :class:`methods.mf.bd.Bd`, :class:`methods.mf.icm.Icm`, :class:`methods.mf.lfnmf.Lfnmf`
                  :class:`methods.mf.lsnmf.Lsnmf`, :class:`methods.mf.nmf.Nmf`, :class:`methods.mf.nsnmf.Nsmf`, :class:`methods.mf.pmf.Pmf`, 
                  :class:`methods.mf.psmf.Psmf`, :class:`methods.mf.snmf.Snmf`, :class:`methods.mf.bmf.Bmf`
    :param n_run: It specifies the number of runs of the algorithm. Default is 1.
    :type n_run: `int`
    :param callback: Pass a callback function that is called after each run when performing multiple runs. This is useful
                     if one wants to save summary measures or process the result before it gets discarded. The callback
                     function is called with only one argument :class:`model.nmf_fit` that contains the fitted model. Default is None.
    :type callback: `function`
    :param initialize_only: The specified MF model and its parameters will only be initialized. Factorization will not
                            run. Default is False.
    :type initialize_only: `bool`
    :param options: Specify some runtime or algorithm specific options. For details on algorithm specific options see specific algorithm
                    documentation. Runtime specific options are:
                    #. When option track=True is specified, the fitted factorization model is tracked during the multiple runs of the algorithm. 
                       This option is taken into account only when multiple runs are executed (:param:`n_run` > 1). From each run of the 
                       factorization all matrix factors are retained, which can be very space consuming. In that case setting the callback 
                       function with :param:`callback` is advised which is executed after each run. Tracking is useful for performing some
                       quality or performance measures (e.g. cophenetic correlation, consensus matrix, dispersion).
                    Default is None. 
    :type options: option specific
    
     Stopping criteria:
     If multiple criteria are passed, the satisfiability of one terminates the factorization run. 

    :param max_iter: Maximum number of factorization iterations. Note that the number of iterations depends
                on the speed of method convergence. Default is 30.
    :type max_iter: `int`
    :param min_residuals: Minimal required improvement of the residuals from the previous iteration. They are computed 
                between the target matrix and its MF estimate using the objective function associated to the MF algorithm. 
                Default is None.
    :type min_residuals: `float` 
    :param test_conv: It indicates how often convergence test is done. By default convergence is tested each iteration. 
    :type: `int`
    """
    if seed.__str__().lower() not in l_seed:
        raise utils.MFError("Unrecognized seeding method. Choose from: %s" % ", ".join(l_seed))
    if method.__str__().lower() not in l_factorization: 
        raise utils.MFError("Unrecognized MF method. Choose from: %s" % ", ".join(l_factorization))
    mf_model = None
    try:
        if type(method) is str:
            mf_model = methods.factorization.methods[method.lower()](V = target, seed = seed, W = W, H = H, rank = rank,
                     max_iter = max_iter, min_residuals = min_residuals, test_conv = test_conv,
                     n_run = n_run, callback = callback, options = options)
        else:
            mf_model = method(V = target, seed = seed, W = W, H = H, rank = rank,
                     max_iter = max_iter, min_residuals = min_residuals, test_conv = test_conv,
                     n_run = n_run, callback = callback, options = options)
    except Exception as str_error:
        raise utils.MFError("Model initialization has been unsuccessful: " + str(str_error))
    if not initialize_only:
        return mf_model.run()
    else:
        return mf_model
    
def mf_run(mf_model):
    """
    Run the specified MF algorithm.
    
    Return fitted factorization model storing MF results. 
    
    :param mf_model: The underlying initialized model of matrix factorization.
    :type mf_model: Class inheriting :class:`models.nmf.Nmf`
    """
    if mf_model.__str__() not in l_factorization:
        raise utils.MFError("Unrecognized MF method.")
    return mf_model.run()

