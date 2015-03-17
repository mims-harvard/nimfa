
# Example call of SNMNMF with algorithm specific parameters set
snmnmf = nimfa.Snmnmf(V=V, V1=V1,
                seed="random_c",
                rank=10,
                max_iter=12,
                A=sp.rand(V1.shape[1], V1.shape[1], density=0.7, format='csr'),
                B=sp.rand(V.shape[1], V1.shape[1], density=0.7, format='csr'),
                gamma=0.01,
                gamma_1=0.01,
                lamb=0.01,
                lamb_1=0.01)
snmnmf_fit = snmnmf()


# Example call of BD with algorithm specific parameters set
bd = nimfa.Bd(V,
                seed="random_c",
                rank=10,
                max_iter=12,
                alpha=np.zeros((V.shape[0], 10)),
                beta=np.zeros((10, V.shape[1])),
                theta=.0,
                k=.0,
                sigma=1.,
                skip=100,
                stride=1,
                n_w=np.zeros((10, 1)),
                n_h=np.zeros((10, 1)),
                n_sigma=False)
bd_fit = bd()


# Example call of BMF with algorithm specific parameters set
bmf = nimfa.Bmf(V,
                seed="nndsvd",
                rank=10,
                max_iter=12,
                lambda_w=1.1,
                lambda_h=1.1)
bmf_fit = bmf()


# Example call of ICM with algorithm specific parameters set
icm = nimfa.Icm(V,
                seed="nndsvd",
                rank=10,
                max_iter=12,
                iiter=20,
                alpha=np.random.randn(V.shape[0], 10),
                beta=np.random.randn(10, V.shape[1]),
                theta=0.,
                k=0.,
                sigma=1.)
icm_fit = icm()


# Example call of LFNMF with algorithm specific parameters set
lfnmf = nimfa.Lfnmf(V,
                seed=None,
                W=np.random.rand(V.shape[0], 10),
                H=np.random.rand(10, V.shape[1]),
                rank=10,
                max_iter=12,
                alpha=0.01)
lfnmf_fit = lfnmf()


# Example call of LSNMF with algorithm specific parameters set
lsnmf = nimfa.Lsnmf(V,
                seed="random_vcol",
                rank=10,
                max_iter=12,
                sub_iter=10,
                inner_sub_iter=10,
                beta=0.1)
lsnmf_fit = lsnmf()


# Example call of NMF - Euclidean with algorithm specific parameters set
nmf = nimfa.Nmf(V,
                seed="nndsvd",
                rank=10,
                max_iter=12,
                update='euclidean',
                objective='fro')
nmf_fit = nmf()


# Example call of NMF - Divergence with algorithm specific parameters set
nmf = nimfa.Nmf(V,
                seed="random_c",
                rank=10,
                max_iter=12,
                update='divergence',
                objective='div')
nmf_fit = nmf()


# Example call of NMF - Connectivity with algorithm specific parameters set
nmf = nimfa.Nmf(V,
                rank=10,
                seed="random_vcol",
                max_iter=200,
                update='euclidean',
                objective='conn',
                conn_change=40)
nmf_fit = nmf()


# Example call of NSNMF with algorithm specific parameters set
nsnmf = nimfa.Nsnmf(V,
                seed="random",
                rank=10,
                max_iter=12,
                theta=0.5)
nsnmf_fit = nsnmf()


# Example call of PMF with algorithm specific parameters set
pmf = nimfa.Pmf(V,
                seed="random_vcol",
                rank=10,
                max_iter=12,
                rel_error=1e-5)
pmf_fit = pmf()


# Example call of PSMF with algorithm specific parameters set
psmf = nimfa.Psmf(V,
                seed=None,
                rank=10,
                max_iter=12,
                prior=np.random.rand(10))
psmf_fit = psmf()


# Example call of SNMF/R with algorithm specific parameters set
snmf = nimfa.Snmf(V,
                seed="random_c",
                rank=10,
                max_iter=12,
                version='r',
                eta=1.,
                beta=1e-4,
                i_conv=10,
                w_min_change=0)
snmf_fit = snmf()


# Example call of SNMF/L with algorithm specific parameters set
snmf = nimfa.Snmf(V,
                seed="random_vcol",
                rank=10,
                max_iter=12,
                version='l',
                eta=1.,
                beta=1e-4,
                i_conv=10,
                w_min_change=0)
snmf_fit = snmf()

# Example call of PMFCC with algorithm specific parameters set
pmfcc = nimfa.Pmfcc(V,
                seed="random_vcol",
                rank=10,
                max_iter=30,
                theta=np.random.rand(V.shape[1], V.shape[1]))
pmfcc_fit = pmfcc()
