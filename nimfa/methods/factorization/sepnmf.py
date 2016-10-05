
"""
#########################################
SepNmf (``methods.factorization.sepnmf``)
#########################################

**Separable Nonnegative Matrix Factorization (SepNMF)** [Damle2014]_, [Benson2014]_,
[Kumar2013]_, [Gillis2014]_, [Tepper2015]_, [Kapralov2016]_

Separable NMF was introduced by Donoho and Stodden (2003) and polynomial time algorithms were given by Arora et al 2012.
Other algorithms such as XRAY [Kumar2013]_, SPA [Gillis2014]_ and more recently SC
[Tepper2015]_ and CG [Kapralov2016]_ have been proposed.

SepNMF can be used for problems which satisfy the ``pure-pixel'' assumption which occurs in
hyper-spectral imaging and document analysis settings.

"""
try:
    from future_builtins import zip
except ImportError: # not 2.6+ or is 3.x
    try:
        from itertools import izip as zip # < 2.5 or 3.x
    except ImportError:
        pass
import collections
import scipy.optimize as optimize
import scipy.sparse as sp
from nimfa.models import *
from nimfa.utils.linalg import *
from nimfa.utils.utils import *

__all__ = ['SepNmf']


class SepNmf(nmf_std.Nmf_std):
    """
    :param V: The target matrix to estimate.
    :type V: Instance of the :class:`scipy.sparse` sparse matrices types,
       :class:`numpy.ndarray`, :class:`numpy.matrix` or tuple of instances of
       the latter classes.

    :param rank: The factorization rank to achieve. Default is 30.
    :type rank: `int`

    :param n_run: It specifies the number of runs of the algorithm. Default is
       1. If multiple runs are performed, fitted factorization model with the
       lowest objective function value is retained.
    :type n_run: `int`

    :param callback: Pass a callback function that is called after each run when
       performing multiple runs. This is useful if one wants to save summary
       measures or process the result before it gets discarded. The callback
       function is called with only one argument :class:`models.mf_fit.Mf_fit` that
       contains the fitted model. Default is None.
    :type callback: `function`

    :param callback_init: Pass a callback function that is called after each
       initialization of the matrix factors. In case of multiple runs the function
       is called before each run (more precisely after initialization and before
       the factorization of each run). In case of single run, the passed callback
       function is called after the only initialization of the matrix factors.
       This is useful if one wants to obtain the initialized matrix factors for
       further analysis or additional info about initialized factorization model.
       The callback function is called with only one argument
       :class:`models.mf_fit.Mf_fit` that (among others) contains also initialized
       matrix factors. Default is None.
    :type callback_init: `function`

    :param track_factor: When :param:`track_factor` is specified, the fitted
        factorization model is tracked during multiple runs of the algorithm. This
        option is taken into account only when multiple runs are executed
        (:param:`n_run` > 1). From each run of the factorization all matrix factors
        are retained, which can be very space consuming. If space is the problem
        setting the callback function with :param:`callback` is advised which is
        executed after each run. Tracking is useful for performing some quality or
        performance measures (e.g. cophenetic correlation, consensus matrix,
        dispersion). By default fitted model is not tracked.
    :type track_factor: `bool`

    :param track_error: Tracking the residuals error. Only the residuals from
        each iteration of the factorization are retained. Error tracking is not
        space consuming. By default residuals are not tracked and only the final
        residuals are saved. It can be used for plotting the trajectory of the
        residuals.
    :type track_error: `bool`
    """
    def __init__(self, V, rank=30, test_conv=None, n_run=1, callback=None,
                 callback_init=None, track_factor=False, track_error=False,
                 compression=None, selection='spa', **options):
        params = vars()
        del params['self']
        params.update({
            'name': 'sepnmf',
            'W': None,
            'H': None,
            'seed': None,
            'aseeds': 'none',
        })
        super(SepNmf, self).__init__(params)

        if compression != 'count_gauss' and selection == 'none':
            raise MFError('Trivial selection only works with count_gauss '
                          'compression')
        if self.compression is not None:
            self.compression = getattr(self, self.compression)
        self.selection = getattr(self, self.selection)
        if self.track_factor and self.n_run > 1 or self.track_error:
            self.tracker = mf_track.Mf_track()
        else:
            self.tracker = None

    def factorize(self):
        """
        Compute matrix factorization.

        Return fitted factorization model.
        """
        for run in range(self.n_run):
            self.H = zeros_for(self.V, (self.rank, self.V.shape[1]))
            if self.compression:
                V = self.compression()
            else:
                V = self.V
            cols = self.selection(V)
            self.H, obj = nnls(V, V[:, cols])
            self.W = self.V[:, cols]

            if self.callback:
                self.final_obj = obj
                self.n_iter = iter
                mffit = mf_fit.Mf_fit(self)
                self.callback(mffit)
            if self.track_factor:
                self.tracker.track_factor(run, W=self.W, H=self.H,
                                          final_obj=obj, n_iter=0)
            # if multiple runs are performed, fitted factorization model with
            # the lowest objective function value is retained
            if run == 0 or obj <= best_obj:
                best_obj = obj
                self.n_iter = 0
                self.final_obj = obj
                mffit = mf_fit.Mf_fit(copy.deepcopy(self))

        mffit.fit.tracker = self.tracker
        return mffit

    def __str__(self):
        str_format = '{0} - compression: {1} - selection: {2}'
        comp = self.compression.__name__ if self.compression else 'none'
        return str_format.format(self.name, comp, self.selection.__name__)

    def __repr__(self):
        return self.name

    def count_gauss(self, oversampling_factor=5):
        """
        Project the columns of the matrix V into the lower dimension
        new_dim using count sketch + gaussian algorithm
        """
        old_dim = self.V.shape[0]

        # ksq = new_dim * new_dim
        ksq = self.rank * oversampling_factor  # was not converging for scree plots of damle/sun

        R = np.random.randint(ksq, size=old_dim)
        C = np.arange(old_dim)
        D = np.random.choice([-1, 1], size=old_dim)
        S = scipy.sparse.csr_matrix((D, (R, C)), shape=(ksq, old_dim))

        G = np.random.randn(self.rank, ksq)
        M_red = dot(G, dot(S, self.V))
        return M_red.todense() / np.sqrt(self.rank)

    def structured(self, n_power_iter=0, oversampling=10, min_comp=20):
        """
        Structured compression algorithm
        """
        n = self.V.shape[1]
        comp_level = min(max(min_comp, self.rank + oversampling), n)
        omega = np.random.standard_normal(size=(n, comp_level))

        mat_h = self.V.dot(omega)
        for _ in range(n_power_iter):
            mat_h = self.V.dot(self.V.T.dot(mat_h))
        q, _ = np.linalg.qr(mat_h)
        return q.T.dot(self.V)

    def qr(self):
        """
        QR compression algorithm
        """
        q, _ = np.linalg.qr(self.V)
        return q.T.dot(self.V)

    def spa(self, V):
        """
        Successive projection algorithm (SPA) algorithm for extreme
        column selection in separable NMF.
        :param V: The data matrix.
        :type V: Instance of the :class:`scipy.sparse` sparse matrices
        types, :class:`numpy.ndarray`, or :class:`numpy.matrix`.
        :return: The indices of the columns chosen by SPA.
        :rtype: List of ints.
        """
        colnorms = norm_axis(V, p=1, axis=0)
        x = elop(V, colnorms, div)
        cols = []
        m, n = x.shape
        for _ in range(self.rank):
            col_norms = norm_axis(x, axis=0)
            col_norms[0, cols] = -1
            _, col_ind = argmax(col_norms)
            cols.append(col_ind)
            col = x[:, col_ind]  # col is a column vector
            x = dot(np.eye(m) - dot(col, col.T) / col_norms[0, col_ind], x)
        return cols

    def xray(self, V):
        """
        X-ray algorithm for extreme column selection in separable NMF.
        :param V: The data matrix.
        :type V: Instance of the :class:`scipy.sparse` sparse matrices
        types, :class:`numpy.ndarray`, or :class:`numpy.matrix`.
        :return: The indices of the columns chosen by X-ray.
        :rtype: List of ints.
        """
        cols = []
        R = V
        while len(cols) < self.rank:
            # Loop until we choose a column that has not been selected.
            while True:
                p = np.random.random((1, V.shape[0]))
                scores = norm_axis(dot(R.T, V), axis=0)
                scores = elop(scores, dot(p, V), div)
                scores[0, cols] = -1  # IMPORTANT
                best_col = np.argmax(scores)
                if best_col in cols:
                    # Re-try
                    continue
                else:
                    cols.append(best_col)
                    H, _ = nnls(V, V[:, cols])
                    R = V - dot(V[:, cols], H)
                    break
        return cols

    def none(self, V):
        """
        Trivial algorithm for extreme column selection in separable NMF.
        :param V: The data matrix.
        :type V: Instance of the :class:`scipy.sparse` sparse matrices
        types, :class:`numpy.ndarray`, or :class:`numpy.matrix`.
        :return: The indices of the columns chosen by X-ray.
        :rtype: List of ints.
        """
        idx = collections.Counter()
        idx.update(argmax(V, axis=0)[1].tolist()[0])
        idx.update(argmin(V, axis=0)[1].tolist()[0])
        return next(zip(*idx.most_common(self.rank)))


def objective(V, W, H):
    """
    Compute squared Frobenius norm of a target matrix and its NMF
    estimate.
    """
    R = V - dot(W, H)
    return multiply(R, R).sum()


def nnls(V, W):
    """
    Compute H, the coefficient matrix, by nonnegative least squares
    to minimize the Frobenius norm.  Given the data matrix V and the
    basis matrix W, H is
    .. math:: \arg\min_{Y \ge 0} \| V - W H \|_F.
    :param V: The data matrix.
    :type V: numpy.ndarray
    :param W: The data matrix.
    :type W: numpy.ndarray
    :return: The matrix H and the relative residual.
    """
    ncols = V.shape[1]
    H = []
    total_res = 0
    for i in range(ncols):
        b = np.squeeze(np.array(V[:, i]))
        sol, res = optimize.nnls(np.array(W), b)
        H.append(sol[:, np.newaxis])
        total_res += res ** 2
    return np.asmatrix(hstack(H)), total_res


def zeros_for(mat, size):
    if sp.isspmatrix(mat):
        format_dict = {'bsr': sp.bsr_matrix, 'coo': sp.coo_matrix,
                       'csc': sp.csc_matrix, 'csr': sp.csr_matrix,
                       'dia': sp.dia_matrix, 'dok': sp.dok_matrix,
                       'lil': sp.lil_matrix}
        cls = format_dict[format]
        return cls(size)
    else:
        return np.zeros(size)


def norm_axis(X, p=None, axis=None):
    """
    Compute entry-wise norms (! not induced/operator norms).

    :param X: The input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    :param p: Order of the norm.
    :type p: `str` or `float`
    """
    assert 1 in X.shape or p != 2 or axis is not None,\
        "Computing entry-wise norms only."
    if sp.isspmatrix(X):
        return sla.norm(X, ord=p, axis=axis)
    else:
        n = nla.norm(np.mat(X), ord=p, axis=axis)
        if axis is None:
            return n
        else:
            return np.asmatrix(n)
