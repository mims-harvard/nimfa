
# Example call of SNMNMF with algorithm specific parameters set
fctr = nimfa.mf(target = (V, V1), 
              seed = "random_c", 
              rank = 10, 
              method = "snmnmf", 
              max_iter = 12, 
              initialize_only = True,
              A = abs(sp.rand(V1.shape[1], V1.shape[1], density = 0.7, format = 'csr')),
              B = abs(sp.rand(V.shape[1], V1.shape[1], density = 0.7, format = 'csr')), 
              gamma = 0.01,
              gamma_1 = 0.01,
              lamb = 0.01,
              lamb_1 = 0.01)
fctr_res = nimfa.mf_run(fctr)


# Example call of BD with algorithm specific parameters set
fctr = nimfa.mf(V, 
              seed = "random_c", 
              rank = 10, 
              method = "bd", 
              max_iter = 12, 
              initialize_only = True,
              alpha = np.mat(np.zeros((V.shape[0], rank))),
              beta = np.mat(np.zeros((rank, V.shape[1]))),
              theta = .0,
              k = .0,
              sigma = 1., 
              skip = 100,
              stride = 1,
              n_w = np.mat(np.zeros((rank, 1))),
              n_h = np.mat(np.zeros((rank, 1))),
              n_sigma = False)
fctr_res = nimfa.mf_run(fctr)


# Example call of BMF with algorithm specific parameters set
fctr = nimfa.mf(V, 
              seed = "nndsvd", 
              rank = 10, 
              method = "bmf", 
              max_iter = 12, 
              initialize_only = True,
              lambda_w = 1.1,
              lambda_h = 1.1)
fctr_res = nimfa.mf_run(fctr)


# Example call of ICM with algorithm specific parameters set    
fctr = nimfa.mf(V, 
              seed = "nndsvd", 
              rank = 10, 
              method = "icm", 
              max_iter = 12, 
              initialize_only = True,
              iiter = 20,
              alpha = pnrg.randn(V.shape[0], rank),
              beta = pnrg.randn(rank, V.shape[1]), 
              theta = 0.,
              k = 0.,
              sigma = 1.)
fctr_res = nimfa.mf_run(fctr)


# Example call of LFNMF with algorithm specific parameters set    
fctr = nimfa.mf(V, 
              seed = None,
              W = abs(pnrg.randn(V.shape[0], rank)), 
              H = abs(pnrg.randn(rank, V.shape[1])),
              rank = 10, 
              method = "lfnmf", 
              max_iter = 12, 
              initialize_only = True,
              alpha = 0.01)
fctr_res = nimfa.mf_run(fctr)
    

# Example call of LSNMF with algorithm specific parameters set    
fctr = nimfa.mf(V, 
              seed = "random_vcol", 
              rank = 10, 
              method = "lsnmf", 
              max_iter = 12, 
              initialize_only = True,
              sub_iter = 10,
              inner_sub_iter = 10, 
              beta = 0.1)
fctr_res = nimfa.mf_run(fctr)



# Example call of NMF - Euclidean with algorithm specific parameters set
fctr = nimfa.mf(V, 
              seed = "nndsvd", 
              rank = 10, 
              method = "nmf", 
              max_iter = 12, 
              initialize_only = True,
              update = 'euclidean',
              objective = 'fro')
fctr_res = nimfa.mf_run(fctr)


# Example call of NMF - Divergence with algorithm specific parameters set
fctr = nimfa.mf(V, 
              seed = "random_c", 
              rank = 10, 
              method = "nmf", 
              max_iter = 12, 
              initialize_only = True,
              update = 'divergence',
              objective = 'div')
fctr_res = nimfa.mf_run(fctr)


# Example call of NMF - Connectivity with algorithm specific parameters set
fctr = nimfa.mf(V, 
             method = "nmf", 
             rank = 10, 
             seed = "random_vcol", 
             max_iter = 200, 
             update = 'euclidean', 
             objective = 'conn',
             conn_change = 40,
             initialize_only = True)
fctr_res = nimfa.mf_run(fctr)
    
    
# Example call of NSNMF with algorithm specific parameters set    
fctr = nimfa.mf(V, 
              seed = "random", 
              rank = 10, 
              method = "nsnmf", 
              max_iter = 12, 
              initialize_only = True,
              theta = 0.5)
fctr_res = nimfa.mf_run(fctr)
    
    
# Example call of PMF with algorithm specific parameters set    
fctr = nimfa.mf(V, 
              seed = "random_vcol", 
              rank = 10, 
              method = "pmf", 
              max_iter = 12, 
              initialize_only = True,
              rel_error = 1e-5)
fctr_res = nimfa.mf_run(fctr)


# Example call of PSMF with algorithm specific parameters set    
fctr = nimfa.mf(V, 
              seed = None,
              rank = 10, 
              method = "psmf", 
              max_iter = 12, 
              initialize_only = True,
              prior = prng.uniform(low = 0., high = 1., size = 10))
fctr_res = nimfa.mf_run(fctr)


# Example call of SNMF/R with algorithm specific parameters set
fctr = nimfa.mf(V, 
              seed = "random_c", 
              rank = 10, 
              method = "snmf", 
              max_iter = 12, 
              initialize_only = True,
              version = 'r',
              eta = 1.,
              beta = 1e-4, 
              i_conv = 10,
              w_min_change = 0)
fctr_res = nimfa.mf_run(fctr)

    
# Example call of SNMF/L with algorithm specific parameters set    
fctr = nimfa.mf(V, 
              seed = "random_vcol", 
              rank = 10, 
              method = "snmf", 
              max_iter = 12, 
              initialize_only = True,
              version = 'l',
              eta = 1.,
              beta = 1e-4, 
              i_conv = 10,
              w_min_change = 0)
fctr_res = nimfa.mf_run(fctr)

# Example call of PMFCC with algorithm specific parameters set    
fctr = nimfa.mf(V, 
              seed = "random_vcol", 
              rank = 10, 
              method = "pmfcc", 
              max_iter = 30, 
              initialize_only = True,
              theta = np.random.random((V.shape[1], V.shape[1])))
fctr_res = nimfa.mf_run(fctr)

