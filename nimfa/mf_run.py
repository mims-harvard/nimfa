
"""
    ###################
    Mf_run (``mf_run``)
    ###################

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
    #. [mandatory] Choose the seeding method to compute the starting point passed to the algorithm. 
    #. [mandatory] Provide the target object to estimate. 
    #. Provide additional runtime or algorithm specific parameters.
    
"""

from utils import *

import examples
import methods

l_factorization = methods.list_mf_methods()
l_seed = methods.list_seeding_methods()

def mf(target, seed = None, W = None, H = None,  
       rank = 30, method = "nmf",
       max_iter = 30, min_residuals = None, test_conv = None,
       n_run = 1, callback = None, callback_init = None, initialize_only = True, **options):
    """
    Run the specified MF algorithm.
    
    Return fitted factorization model storing MF results. If :param:`initialize_only` is set, only initialized model is returned (default behaviour).
    
    :param target: The target matrix to estimate. Some algorithms (e. g. multiple NMF) specify more than one target matrix. 
                   In that case target matrices are passed as tuples. Internally, additional attributes with names following 
                   Vn pattern are created, where n is the consecutive index of target matrix. Zero index is omitted 
                   (there are V, V1, V2, V3, etc. matrices and then H, H1, H2, etc. and W, W1, W2, etc. respectively - depends
                   on the algorithm).
    :type target: Instance of the :class:`scipy.sparse` sparse matrices types, :class:`numpy.ndarray`, :class:`numpy.matrix` or
                  tuple of instances of the latter classes.
    :param seed: Specify method to seed the computation of a factorization. If specified :param:`W` and :param:`H` seeding 
                 must be None. If neither seeding method or initial fixed factorization is specified, random initialization is used
    :type seed: `str` naming the method or :class:`methods.seeding.nndsvd.Nndsvd` or None
    :param W: Specify initial factorization of basis matrix W. Default is None. When specified, :param:`seed` must be None.
    :type W: :class:`scipy.sparse` or :class:`numpy.ndarray` or :class:`numpy.matrix` or None
    :param H: Specify initial factorization of mixture matrix H. In case of factorizations with multiple MF underlying model, initialization 
              of multiple mixture matrices can be passed as tuples (in order H, H1, H2, etc. respectively). Default is None. When 
              specified, :param:`seed` must be None.
    :type H: Instance of the :class:`scipy.sparse` sparse matrices types, :class:`numpy.ndarray`, :class:`numpy.matrix`,
             tuple of instances of the latter classes or None
    :param rank: The factorization rank to achieve. Default is 30.
    :type rank: `int`
    :param method: The algorithm to use to perform MF on target matrix. Default is :class:`methods.factorization.nmf.Nmf`
    :type method: `str` naming the algorithm or :class:`methods.factorization.bd.Bd`, 
                  :class:`methods.factorization.icm.Icm`, :class:`methods.factorization.Lfnmf.Lfnmf`
                  :class:`methods.factorization.lsnmf.Lsnmf`, :class:`methods.factorization.nmf.Nmf`, 
                  :class:`methods.factorization.nsnmf.Nsmf`, :class:`methods.factorization.pmf.Pmf`, 
                  :class:`methods.factorization.psmf.Psmf`, :class:`methods.factorization.snmf.Snmf`, 
                  :class:`methods.factorization.bmf.Bmf`, :class:`methods.factorization.snmnmf.Snmnmf`
    :param n_run: It specifies the number of runs of the algorithm. Default is 1. If multiple runs are performed, fitted factorization
                  model with the lowest objective function value is retained. 
    :type n_run: `int`
    :param callback: Pass a callback function that is called after each run when performing multiple runs. This is useful
                     if one wants to save summary measures or process the result before it gets discarded. The callback
                     function is called with only one argument :class:`models.mf_fit.Mf_fit` that contains the fitted model. Default is None.
    :type callback: `function`
    :param callback_init: Pass a callback function that is called after each initialization of the matrix factors. In case of multiple runs
                          the function is called before each run (more precisely after initialization and before the factorization of each run). In case
                          of single run, the passed callback function is called after the only initialization of the matrix factors. This is 
                          useful if one wants to obtain the initialized matrix factors for further analysis or additional info about initialized
                          factorization model. The callback function is called with only one argument :class:`models.mf_fit.Mf_fit` that (among others) 
                          contains also initialized matrix factors. Default is None. 
    :type callback_init: `function`
    :param initialize_only: The specified MF model and its parameters will only be initialized. Model initialization includes:
                                #. target matrix format checking and possibly conversion into one of accepting formats,
                                #. checking if target matrix (or matrices) are nonnegative (in case of NMF factorization algorithms),
                                #. validation of the specified factorization method,
                                #. validation of the specified initialization method. 
                            When this parameter is specified factorization will not be ran. Default is True.
    :type initialize_only: `bool`
    
    
    **Runtime specific parameters**
    
    In addition to general parameters above, there is a possibility to specify runtime specific and factorization algorithm
    specific options. 
    
    .. note:: For details on algorithm specific options see specific algorithm documentation.
                               
    The following are runtime specific options.
                    
     :param track_factor: When :param:`track_factor` is specified, the fitted factorization model is tracked during multiple
                        runs of the algorithm. This option is taken into account only when multiple runs are executed 
                        (:param:`n_run` > 1). From each run of the factorization all matrix factors are retained, which 
                        can be very space consuming. If space is the problem setting the callback function with :param:`callback` 
                        is advised which is executed after each run. Tracking is useful for performing some quality or 
                        performance measures (e.g. cophenetic correlation, consensus matrix, dispersion). By default fitted model
                        is not tracked.
     :type track_factor: `bool`
     :param track_error: Tracking the residuals error. Only the residuals from each iteration of the factorization are retained. 
                        Error tracking is not space consuming. By default residuals are not tracked and only the final residuals
                        are saved. It can be used for plotting the trajectory of the residuals.
     :type track_error: `bool`
    
    
    **Stopping criteria parameters**
     
    If multiple criteria are passed, the satisfiability of one terminates the factorization run. 
    
    .. note:: Some factorization and initialization methods have beside the following also algorithm specific
              stopping criteria. For these details see specific algorithm's documentation.

    :param max_iter: Maximum number of factorization iterations. Note that the number of iterations depends
                on the speed of method convergence. Default is 30.
    :type max_iter: `int`
    :param min_residuals: Minimal required improvement of the residuals from the previous iteration. They are computed 
                between the target matrix and its MF estimate using the objective function associated to the MF algorithm. 
                Default is None.
    :type min_residuals: `float` 
    :param test_conv: It indicates how often convergence test is done. By default convergence is tested each iteration. 
    :type test_conv: `int`
    """
    if seed.__str__().lower() not in l_seed:
        raise utils.MFError("Unrecognized seeding method. Choose from: %s" % ", ".join(l_seed))
    if method.__str__().lower() not in l_factorization: 
        raise utils.MFError("Unrecognized MF method. Choose from: %s" % ", ".join(l_factorization))
    mf_model = None
    # Construct factorization model
    try:
        if isinstance(method, str):
            mf_model = methods.factorization.methods[method.lower()](V = target, seed = seed, W = W, H = H, H1 = None, 
                     rank = rank, max_iter = max_iter, min_residuals = min_residuals, test_conv = test_conv,
                     n_run = n_run, callback = callback, callback_init = callback_init, options = options)
        else:
            mf_model = method(V = target, seed = seed, W = W, H = H, H1 = None, rank = rank,
                     max_iter = max_iter, min_residuals = min_residuals, test_conv = test_conv,
                     n_run = n_run, callback = callback, callback_init = callback_init, options = options)
    except Exception as str_error:
        raise utils.MFError("Model initialization has been unsuccessful: " + str(str_error))
    # Check if chosen seeding method is compatible with chosen factorization method or fixed initialization is passed
    _compatibility(mf_model)
    # return factorization model if only initialization was requested
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
    if repr(mf_model) not in l_factorization:
        raise utils.MFError("Unrecognized MF method.")
    return mf_model.run()

def _compatibility(mf_model):
    """
    Check if chosen seeding method is compatible with chosen factorization method or fixed initialization is passed.
    
    :param mf_model: The underlying initialized model of matrix factorization.
    :type mf_model: Class inheriting :class:`models.nmf.Nmf`
    """
    W = mf_model.basis()
    H = mf_model.coef(0)
    H1 = mf_model.coef(1) if mf_model.model_name == 'mm' else None
    if mf_model.seed == None and W == None and H == None and H1 == None: mf_model.seed = None if "none" in mf_model.aseeds else "random"
    if W != None and H != None:
        if mf_model.seed != None and mf_model.seed != "fixed":
            raise utils.MFError("Initial factorization is fixed. Seeding method cannot be used.")
        else:
            mf_model.seed = methods.seeding.fixed.Fixed()
            mf_model.seed._set_fixed(W = W, H = H, H1 = H1)
    __is_smdefined(mf_model)
    __compatibility(mf_model)

def __is_smdefined(mf_model):
    """Check if MF and seeding methods are well defined."""
    if isinstance(mf_model.seed, str):
        if mf_model.seed in methods.seeding.methods:
            mf_model.seed = methods.seeding.methods[mf_model.seed]()
        else: raise utils.MFError("Unrecognized seeding method.")
    else:
        if not str(mf_model.seed).lower() in methods.seeding.methods:
            raise utils.MFError("Unrecognized seeding method.")
         
def __compatibility(mf_model):
    """Check if MF model is compatible with the seeding method."""
    if not str(mf_model.seed).lower() in mf_model.aseeds:
        raise utils.MFError("MF model is incompatible with chosen seeding method.") 
