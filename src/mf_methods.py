
"""
    This module implements the main interface to launch matrix factorization algorithms. 
    MF algorithms can be combined with implemented seeding methods.
    
    Returned object can be directly passed to visualization or comparison utilities or as initialization 
    to another factorization method.
    
    #. [mandatory] Choose the algorithm to perform MF on target matrix.
    #. Choose the MF model.
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
import utils
import models.nmf_std as std

l_mf = methods.list_mf_methods()
l_seed = methods.list_seeding_methods()

def mf(target = None, seed = None, W = None, H = None,  
       rank = 30, method = methods.mf.nmf.Nmf,
       max_iters = None, min_residuals = None, 
       n_run = 1, model = std.Nmf_std, callback = None, initialize_only = False, **options):
    """
    Run the specified MF algorithm.
    
    :param target: The target matrix to estimate.
    :type target: One of the :class:`scipy.sparse` sparse matrices types or :class:`numpy.ndarray` or :class:`numpy.matrix` 
    :param seed: Specify method to seed the computation of a factorization. If specified :param:`W` and :param:`H` must be None.
    :type seed: `str` naming the method or :class:`methods.seeding.nndsvd.Nndsvd` or None
    :param W: Specify initial factorization of basis matrix W. Default is None. When specified, :param:`seed` must be None.
    :type W: :class:`scipy.sparse` or :class:`numpy.ndarray` or :class:`numpy.matrix` or None
    :param H: Specify initial factorization of mixture matrix H. Default is None. When specified, :param:`seed` must be None.
    :type H: :class:`scipy.sparse` or :class:`numpy.ndarray` or :class:`numpy.matrix` or None
    :param rank: The factorization rank to achieve. Default is 30.
    :type rank: `int`
    :param method: The algorithm to use to perform MF on target matrix. Default is :class:`methods.mf.nmf`
    :type method: `str` naming the algorithm or :class:`methods.mf.bd.Bd`, :class:`methods.mf.icm.Icm`, :class:`methods.mf.lnmf.Lnmf`
                :class:`methods.mf.lsnmf.Lsnmf`, :class:`methods.mf.nmf.Nmf`, :class:`methods.mf.nsnmf.Nsmf`, :class:`methods.mf.pmf.Pmf`, 
                :class:`methods.mf.psmf.Psmf`, :class:`methods.mf.snmf.Snmf`
    :param n_run: It specifies the number of runs of the algorithm. Default is 1.
    :type n_run: `int`
    :param model: If not specified in the call, the standard MF model :class:`models.nmf_std.Nmf_std` is used. Some MF algorithms
                have different underlying models, such as nonsmooth NMF, which uses an extra matrix factor.
    :type model: `str` naming the model or :class:`models.nmf_std.Nmf_std`
    :param callback: Pass a callback function that is called after each run when performing multiple runs. This is useful
                if one wants to save summary measures or process the result before it gets discarded. The callback
                function is called with only one argument :class:`model.nmf_fit` that contains the fitted model. Default is None.
    :type callback: `function`
    :param initialize_only: If specified the MF model and its parameters will be only initialized. Factorization will not
                run. Default is False.
    :type initialize_only: `bool`
    :param options: Specify some runtime or algorithm specific options. Default is None.
    :type options: `dict`
    
     Stopping criteria:
     If multiple criteria are passed, the satisfiability of one terminates the factorization run. 

    :param max_iters: Maximum number of factorization iterations. When not specified, the number of iterations depends
                on the speed of method convergence. Default is None
    :type max_iters: `int`
    :param min_residuals: Minimal required improvement of the residuals from the previous iteration. They are computed 
                between the target matrix and its MF estimate using the objective function associated to the MF algorithm. 
                Default is None.
    :type min_residuals: `float` 
    """
    if seed not in l_seed or None and seed.name not in l_seed:
        raise utils.utils.MFError("Unrecognized seeding method. Choose from: %s" % ", ".join(l_seed))
    if method not in l_mf or None and method.name not in l_mf:
        raise utils.utils.MFError("Unrecognized MF method. Choose from: %s" % ", ".join(l_mf))
    mf_model = model(V = target, seed = seed, W = W, H = H,  
                     rank = rank, method = method,
                     max_iters = max_iters, min_residuals = min_residuals, 
                     n_run = n_run, callback = callback, options = options)
    if not initialize_only:
        res = mf_model.run()
    return res