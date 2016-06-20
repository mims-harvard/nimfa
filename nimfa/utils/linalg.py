"""
    #########################
    Linalg (``utils.linalg``)
    #########################
    
    Linear algebra helper routines and wrapper functions for handling sparse
    matrices and dense matrices representation.
"""

import sys
import copy
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import numpy.linalg as nla
from operator import mul, eq, ne, add, ge, le, itemgetter
from operator import truediv as div

from math import sqrt, log, isnan, ceil
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.special import erfc, erfcinv
import warnings

#
# Wrapper functions for handling sparse matrices and dense matrices representation.
###    scipy.sparse, numpy.matrix
#


def diff(X):
    """
    Compute differences between adjacent elements of X.

    :param X: Vector for which consecutive differences are computed.
    :type X: :class:`numpy.matrix`
    """
    assert 1 in X.shape, "sX should be a vector."
    assert not sp.isspmatrix(X), "X is sparse matrix."
    X = X.flatten()
    return [X[0, j + 1] - X[0, j] for j in range(X.shape[1] - 1)]


def sub2ind(shape, row_sub, col_sub):
    """
    Return the linear index equivalents to the row and column subscripts for
    given matrix shape.

    :param shape: Preferred matrix shape for subscripts conversion.
    :type shape: `tuple`
    :param row_sub: Row subscripts.
    :type row_sub: `list`
    :param col_sub: Column subscripts.
    :type col_sub: `list`
    """
    assert len(row_sub) == len(
        col_sub), "Row and column subscripts do not match."
    res = [j * shape[0] + i for i, j in zip(row_sub, col_sub)]
    return res


def trace(X):
    """
    Return trace of sparse or dense square matrix X.

    :param X: Target matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil,
    dia or :class:`numpy.matrix`
    """
    assert X.shape[0] == X.shape[1], "X should be square matrix."
    if sp.isspmatrix(X):
        return sum(X[i, i] for i in range(X.shape[0]))
    else:
        return np.trace(np.mat(X))


def any(X, axis=None):
    """
    Test whether any element along a given axis of sparse or dense matrix X is nonzero.

    :param X: Target matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil,
    dia or :class:`numpy.matrix`
    :param axis: Specified axis along which nonzero test is performed.
    If :param:`axis` not specified, whole matrix is considered.
    :type axis: `int`
    """
    if sp.isspmatrix(X):
        X = X.tocsr()
        assert axis == 0 or axis == 1 or axis is None, "Incorrect axis number."
        if axis is None:
            return len(X.data) != X.shape[0] * X.shape[1]
        res = [0 for _ in range(X.shape[1 - axis])]

        def _caxis(now, row, col):
            res[col] += 1

        def _raxis(now, row, col):
            res[row] += 1
        check = _caxis if axis == 0 else _raxis
        now = 0
        for row in range(X.shape[0]):
            upto = X.indptr[row + 1]
            while now < upto:
                col = X.indices[now]
                check(now, row, col)
                now += 1
        sol = [x != 0 for x in res]
        return np.mat(sol) if axis == 0 else np.mat(sol).T
    else:
        return X.any(axis)


def all(X, axis=None):
    """
    Test whether all elements along a given axis of sparse or dense matrix
    :param:`X` are nonzero.

    :param X: Target matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    :param axis: Specified axis along which nonzero test is performed.
    If :param:`axis` not specified, whole matrix is considered.
    :type axis: `int`
    """
    if sp.isspmatrix(X):
        X = X.tocsr()
        assert axis == 0 or axis == 1 or axis is None, "Incorrect axis number."
        if axis is None:
            return len(X.data) == X.shape[0] * X.shape[1]
        res = [0 for _ in range(X.shape[1 - axis])]

        def _caxis(now, row, col):
            res[col] += 1

        def _raxis(now, row, col):
            res[row] += 1
        check = _caxis if axis == 0 else _raxis
        now = 0
        for row in range(X.shape[0]):
            upto = X.indptr[row + 1]
            while now < upto:
                col = X.indices[now]
                check(now, row, col)
                now += 1
        sol = [x == X.shape[0] if axis == 0 else x == X.shape[1] for x in res]
        return np.mat(sol) if axis == 0 else np.mat(sol).T
    else:
        return X.all(axis)


def find(X):
    """
    Return all nonzero elements indices (linear indices) of sparse or dense
    matrix :param:`X`. It is Matlab notation.

    :param X: Target matrix.
    type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    """
    if sp.isspmatrix(X):
        X = X.tocsr()
        res = []
        now = 0
        for row in range(X.shape[0]):
            upto = X.indptr[row + 1]
            while now < upto:
                col = X.indices[now]
                if X.data[now]:
                    res.append(col * X.shape[0] + row)
                now += 1
        return res
    else:
        return [j * X.shape[0] + i for i in range(X.shape[0]) for j in range(X.shape[1]) if X[i, j]]


def negative(X):
    """
    Check if :param:`X` contains negative elements.

    :param X: Target matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil,
    dia or :class:`numpy.matrix`
    """
    if sp.isspmatrix(X):
        if any(X.data < 0):
            return True
    else:
        if any(np.asmatrix(X) < 0):
            return True


def sort(X):
    """
    Return sorted elements of :param:`X` and array of corresponding
    sorted indices.

    :param X: Target vector.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil,
    dia or :class:`numpy.matrix`
    """
    assert 1 in X.shape, "X should be vector."
    X = X.flatten().tolist()[0]
    return sorted(X), sorted(list(range(len(X))), key=X.__getitem__)


def std(X, axis=None, ddof=0):
    """
    Compute the standard deviation along the specified :param:`axis` of
    matrix :param:`X`.

    :param X: Target matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil,
    dia or :class:`numpy.matrix`
    :param axis: Axis along which deviation is computed. If not specified,
    whole matrix :param:`X` is considered.
    :type axis: `int`
    :param ddof: Means delta degrees of freedom. The divisor used in
    computation is N - :param:`ddof`, where N represents the
    number of elements. Default is 0.
    :type ddof: `float`
    """
    assert len(X.shape) == 2, "Input matrix X should be 2-D."
    assert axis == 0 or axis == 1 or axis is None, "Incorrect axis number."
    if sp.isspmatrix(X):
        if axis is None:
            mean = X.mean()
            no = X.shape[0] * X.shape[1]
            return sqrt(1. / (no - ddof) * sum((x - mean) ** 2 for x in X.data) + (no - len(X.data) * mean ** 2))
        if axis == 0:
            return np.mat([np.std(X[:, i].toarray(), axis, ddof) for i in range(X.shape[1])])
        if axis == 1:
            return np.mat([np.std(X[i, :].toarray(), axis, ddof) for i in range(X.shape[0])]).T
    else:
        return np.std(X, axis=axis, ddof=ddof)


def argmax(X, axis=None):
    """
    Return tuple (values, indices) of the maximum entries of matrix
    :param:`X` along axis :param:`axis`. Row major order.

    :param X: Target matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil,
    dia or :class:`numpy.matrix`
    :param axis: Specify axis along which to operate. If not specified,
    whole matrix :param:`X` is considered.
    :type axis: `int`
    """
    if sp.isspmatrix(X):
        X = X.tocsr()
        assert axis == 0 or axis == 1 or axis is None, "Incorrect axis number."
        res = [[float('-inf'), 0]
               for _ in range(X.shape[1 - axis])] if axis is not None else [float('-inf'), 0]

        def _caxis(row, col):
            if X[row, col] > res[col][0]:
                res[col] = (X[row, col], row)

        def _raxis(row, col):
            if X[row, col] > res[row][0]:
                res[row] = (X[row, col], col)

        def _naxis(row, col):
            if X[row, col] > res[0]:
                res[0] = X[row, col]
                res[1] = row * X.shape[0] + col
        check = _caxis if axis == 0 else _raxis if axis == 1 else _naxis
        [check(row, col) for row in range(X.shape[0])
         for col in range(X.shape[1])]
        if axis is None:
            return res
        elif axis == 0:
            t = list(zip(*res))
            return list(t[0]), np.mat(t[1])
        else:
            t = list(zip(*res))
            return list(t[0]), np.mat(t[1]).T
    else:
        idxX = np.asmatrix(X).argmax(axis)
        if axis is None:
            eX = X[idxX / X.shape[1], idxX % X.shape[1]]
        elif axis == 0:
            eX = [X[idxX[0, idx], col]
                  for idx, col in zip(range(X.shape[1]), range(X.shape[1]))]
        else:
            eX = [X[row, idxX[idx, 0]]
                  for row, idx in zip(range(X.shape[0]), range(X.shape[0]))]
        return eX, idxX


def argmin(X, axis=None):
    """
    Return tuple (values, indices) of the minimum entries of matrix :param:`X`
    along axis :param:`axis`. Row major order.

    :param X: Target matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil,
    dia or :class:`numpy.matrix`
    :param axis: Specify axis along which to operate. If not specified,
    whole matrix :param:`X` is considered.
    :type axis: `int`
    """
    if sp.isspmatrix(X):
        X = X.tocsr()
        assert axis == 0 or axis == 1 or axis is None, "Incorrect axis number."
        res = [[float('inf'), 0]
               for _ in range(X.shape[1 - axis])] if axis is not None else [float('inf'), 0]

        def _caxis(row, col):
            if X[row, col] < res[col][0]:
                res[col] = (X[row, col], row)

        def _raxis(row, col):
            if X[row, col] < res[row][0]:
                res[row] = (X[row, col], col)

        def _naxis(row, col):
            if X[row, col] < res[0]:
                res[0] = X[row, col]
                res[1] = row * X.shape[0] + col
        check = _caxis if axis == 0 else _raxis if axis == 1 else _naxis
        [check(row, col) for row in range(X.shape[0])
         for col in range(X.shape[1])]
        if axis is None:
            return res
        elif axis == 0:
            t = list(zip(*res))
            return list(t[0]), np.mat(t[1])
        else:
            t = list(zip(*res))
            return list(t[0]), np.mat(t[1]).T
    else:
        idxX = np.asmatrix(X).argmin(axis)
        if axis is None:
            eX = X[idxX / X.shape[1], idxX % X.shape[1]]
        elif axis == 0:
            eX = [X[idxX[0, idx], col]
                  for idx, col in zip(range(X.shape[1]), range(X.shape[1]))]
        else:
            eX = [X[row, idxX[idx, 0]]
                  for row, idx in zip(range(X.shape[0]), range(X.shape[0]))]
        return eX, idxX


def repmat(X, m, n):
    """
    Construct matrix consisting of an m-by-n tiling of copies of X.

    :param X: The input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil,
    dia or :class:`numpy.matrix`
    :param m,n: The number of repetitions of :param:`X` along each axis.
    :type m,n: `int`
    """
    if sp.isspmatrix(X):
        return sp.hstack([sp.vstack([X for _ in range(m)], format=X.format) for _ in range(n)], format=X.format)
    else:
        return np.tile(np.asmatrix(X), (m, n))


def inv_svd(X):
    """
    Compute matrix inversion using SVD.

    :param X: The input matrix.
    :type X: :class:`scipy.sparse` or :class:`numpy.matrix`
    """
    U, S, V = svd(X)
    if sp.isspmatrix(S):
        S_inv = _sop_spmatrix(S, op=lambda x: 1. / x)
    else:
        S_inv = np.diag(1. / np.diagonal(S))
    X_inv = dot(dot(V.T, S_inv), U.T)
    return X_inv


def svd(X):
    """
    Compute standard SVD on matrix X.

    :param X: The input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil,
    dia or :class:`numpy.matrix`
    """
    if sp.isspmatrix(X):
        if X.shape[0] <= X.shape[1]:
            U, S, V = _svd_left(X)
        else:
            U, S, V = _svd_right(X)
    else:
        U, S, V = nla.svd(np.mat(X), full_matrices=False)
        S = np.mat(np.diag(S))
    return U, S, V


def _svd_right(X):
    """
    Compute standard SVD on matrix X. Scipy.sparse.linalg.svd ARPACK does
    not allow computation of rank(X) SVD.

    :param X: The input sparse matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    """
    XXt = dot(X, X.T)
    if X.shape[0] > 1:
        if '0.8' in scipy.version.version:
            val, u_vec = sla.eigen_symmetric(XXt, k=X.shape[0] - 1)
        else:
            # In scipy 0.9.0 ARPACK interface has changed. eigen_symmetric
            # routine was renamed to eigsh
            # http://docs.scipy.org/doc/scipy/reference/release.0.9.0.html#scipy-sparse
            try:
                val, u_vec = sla.eigsh(XXt, k=X.shape[0] - 1)
            except sla.ArpackNoConvergence as err:
                # If eigenvalue iteration fails to converge, partially
                # converged results can be accessed
                val = err.eigenvalues
                u_vec = err.eigenvectors
    else:
        val, u_vec = nla.eigh(XXt.todense())
    # remove insignificant eigenvalues
    keep = np.where(val > 1e-7)[0]
    u_vec = u_vec[:, keep]
    val = val[keep]
    # sort eigen vectors (descending)
    idx = np.argsort(val)[::-1]
    val = val[idx]
    # construct U
    U = sp.csr_matrix(u_vec[:, idx])
    # compute S
    tmp_val = np.sqrt(val)
    tmp_l = len(idx)
    S = sp.spdiags(tmp_val, 0, m=tmp_l, n=tmp_l, format='csr')
    # compute V from inverse of S
    inv_S = sp.spdiags(1. / tmp_val, 0, m=tmp_l, n=tmp_l, format='csr')
    V = U.T * X
    V = inv_S * V
    return U, S, V


def _svd_left(X):
    """
    Compute standard SVD on matrix X. Scipy.sparse.linalg.svd ARPACK does
    not allow computation of rank(X) SVD.

    :param X: The input sparse matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    """
    XtX = dot(X.T, X)
    if X.shape[1] > 1:
        if '0.9' in scipy.version.version or '0.10' in scipy.version.version or '0.11' in scipy.version.version:
            # In scipy 0.9.0 ARPACK interface has changed. eigen_symmetric
            # routine was renamed to eigsh
            # http://docs.scipy.org/doc/scipy/reference/release.0.9.0.html#scipy-sparse
            try:
                val, v_vec = sla.eigsh(XtX, k=X.shape[1] - 1)
            except sla.ArpackNoConvergence as err:
                # If eigenvalue iteration fails to converge, partially
                # converged results can be accessed
                val = err.eigenvalues
                v_vec = err.eigenvectors
        else:
            val, v_vec = sla.eigen_symmetric(XtX, k=X.shape[1] - 1)
    else:
        val, v_vec = nla.eigh(XtX.todense())
    # remove insignificant eigenvalues
    keep = np.where(val > 1e-7)[0]
    v_vec = v_vec[:, keep]
    val = val[keep]
    # sort eigen vectors (descending)
    idx = np.argsort(val)[::-1]
    val = val[idx]
    # construct V
    V = sp.csr_matrix(v_vec[:, idx])
    # compute S
    tmp_val = np.sqrt(val)
    tmp_l = len(idx)
    S = sp.spdiags(tmp_val, 0, m=tmp_l, n=tmp_l, format='csr')
    # compute U from inverse of S
    inv_S = sp.spdiags(1. / tmp_val, 0, m=tmp_l, n=tmp_l, format='csr')
    U = X * V * inv_S
    V = V.T
    return U, S, V


def dot(X, Y):
    """
    Compute dot product of matrices :param:`X` and :param:`Y`.

    :param X: First input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    :param Y: Second input matrix.
    :type Y: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    """
    if sp.isspmatrix(X) and sp.isspmatrix(Y):
        return X * Y
    elif sp.isspmatrix(X) or sp.isspmatrix(Y):
        # avoid dense dot product with mixed factors
        return sp.csr_matrix(X) * sp.csr_matrix(Y)
    else:
        return np.asmatrix(X) * np.asmatrix(Y)


def multiply(X, Y):
    """
    Compute element-wise multiplication of matrices :param:`X` and :param:`Y`.

    :param X: First input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    :param Y: Second input matrix.
    :type Y: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    """
    if sp.isspmatrix(X) and sp.isspmatrix(Y):
        return X.multiply(Y)
    elif sp.isspmatrix(X) or sp.isspmatrix(Y):
        return _op_spmatrix(X, Y, np.multiply)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.multiply(np.mat(X), np.mat(Y))


def power(X, s):
    """
    Compute matrix power of matrix :param:`X` for power :param:`s`.

    :param X: Input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    :param s: Power.
    :type s: `int`
    """
    if sp.isspmatrix(X):
        Y = X.tocsr()
        eps = np.finfo(Y.data.dtype).eps if not 'int' in str(
            Y.data.dtype) else 0
        return sp.csr_matrix((np.power(Y.data + eps, s), Y.indices, Y.indptr), Y.shape)
    else:
        eps = np.finfo(X.dtype).eps if not 'int' in str(X.dtype) else 0
        return np.power(X + eps, s)


def sop(X, s=None, op=None):
    """
    Compute scalar element wise operation of matrix :param:`X` and
    scalar :param:`s`.

    :param X: The input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    :param s: Input scalar. If not specified, element wise operation of input
    matrix is computed.
    :type s: `float`
    :param op: Operation to be performed.
    :type op: `func`
    """
    if sp.isspmatrix(X):
        return _sop_spmatrix(X, s, op)
    else:
        return _sop_matrix(X, s, op)


def _sop_spmatrix(X, s=None, op=None):
    """
    Compute sparse scalar element wise operation of matrix X and scalar :param:`s`.

    :param X: The input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    :param s: Input scalar. If not specified, element wise operation of input
    matrix is computed.
    :type s: `float`
    :param op: Operation to be performed.
    :type op: `func`
    """
    R = X.copy().tocsr()
    eps = np.finfo(R.dtype).eps if not 'int' in str(R.dtype) else 0
    now = 0
    for row in range(R.shape[0]):
        upto = R.indptr[row + 1]
        while now < upto:
            R.data[now] = op(R.data[now] + eps, s) if s is not None else op(
                R.data[now] + eps)
            now += 1
    return R


def _sop_matrix(X, s=None, op=None):
    """
    Compute scalar element wise operation of matrix :param:`X` and scalar :param:`s`.

    :param X: The input matrix.
    :type X: :class:`numpy.matrix`
    :param s: Input scalar. If not specified, element wise operation of input
    matrix is computed.
    :type s: `float`
    :param op: Operation to be performed.
    :type op: `func`
    """
    eps = np.finfo(X.dtype).eps if not 'int' in str(X.dtype) else 0
    return op(X + eps, s) if s is not None else op(X + eps)


def elop(X, Y, op):
    """
    Compute element-wise operation of matrix :param:`X` and matrix :param:`Y`.

    :param X: First input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    :param Y: Second input matrix.
    :type Y: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    :param op: Operation to be performed.
    :type op: `func`
    """
    try:
        zp1 = op(0, 1) if sp.isspmatrix(X) else op(1, 0)
        zp2 = op(0, 0)
        zp = zp1 != 0 or zp2 != 0
    except:
        zp = 0
    if sp.isspmatrix(X) or sp.isspmatrix(Y):
        return _op_spmatrix(X, Y, op) if not zp else _op_matrix(X, Y, op)
    else:
        try:
            X[X == 0] = np.finfo(X.dtype).eps
            Y[Y == 0] = np.finfo(Y.dtype).eps
        except ValueError:
            return op(np.mat(X), np.mat(Y))
        return op(np.mat(X), np.mat(Y))


def _op_spmatrix(X, Y, op):
    """
    Compute sparse element-wise operation for operations preserving zeros.

    :param X: First input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    :param Y: Second input matrix.
    :type Y: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    :param op: Operation to be performed.
    :type op: `func`
    """
    # distinction as op is not necessarily commutative
    return __op_spmatrix(X, Y, op) if sp.isspmatrix(X) else __op_spmatrix(Y, X, op)


def __op_spmatrix(X, Y, op):
    """
    Compute sparse element-wise operation for operations preserving zeros.

    :param X: First input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    :param Y: Second input matrix.
    :type Y: :class:`numpy.matrix`
    :param op: Operation to be performed.
    :type op: `func`
    """
    assert X.shape == Y.shape, "Matrices are not aligned."
    eps = np.finfo(Y.dtype).eps if not 'int' in str(Y.dtype) else 0
    Xx = X.tocsr()
    r, c = Xx.nonzero()
    R = op(Xx[r, c], Y[r, c] + eps)
    R = np.array(R)
    assert 1 in R.shape, "Data matrix in sparse should be rank-1."
    R = R[0, :] if R.shape[0] == 1 else R[:, 0]
    return sp.csr_matrix((R, Xx.indices, Xx.indptr), Xx.shape)


def _op_matrix(X, Y, op):
    """
    Compute sparse element-wise operation for operations not preserving zeros.

    :param X: First input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    :param Y: Second input matrix.
    :type Y: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    :param op: Operation to be performed.
    :type op: `func`
    """
    # operation is not necessarily commutative
    assert X.shape == Y.shape, "Matrices are not aligned."
    eps = np.finfo(Y.dtype).eps if not 'int' in str(Y.dtype) else 0
    return np.mat([[op(X[i, j], Y[i, j] + eps) for j in range(X.shape[1])] for i in range(X.shape[0])])


def inf_norm(X):
    """
    Infinity norm of a matrix (maximum absolute row sum).

    :param X: Input matrix.
    :type X: :class:`scipy.sparse.csr_matrix`, :class:`scipy.sparse.csc_matrix`
    or :class:`numpy.matrix`
    """
    if sp.isspmatrix_csr(X) or sp.isspmatrix_csc(X):
        # avoid copying index and ptr arrays
        abs_X = X.__class__(
            (abs(X.data), X.indices, X.indptr), shape=X.shape)
        return (abs_X * np.ones((X.shape[1]), dtype=X.dtype)).max()
    elif sp.isspmatrix(X):
        return (abs(X) * np.ones((X.shape[1]), dtype=X.dtype)).max()
    else:
        return nla.norm(np.asmatrix(X), float('inf'))


def norm(X, p="fro"):
    """
    Compute entry-wise norms (! not induced/operator norms).

    :param X: The input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    :param p: Order of the norm.
    :type p: `str` or `float`
    """
    assert 1 in X.shape or p != 2, "Computing entry-wise norms only."
    if sp.isspmatrix(X):
        fro = lambda X: sum(abs(x) ** 2 for x in X.data) ** (1. / 2)
        inf = lambda X: abs(X).sum(
            axis=1).max() if 1 not in X.shape else abs(X).max()
        m_inf = lambda X: abs(X).sum(
            axis=1).min() if 1 not in X.shape else abs(X).min()
        one = lambda X: abs(X).sum(axis=0).max() if 1 not in X.shape else sum(
            abs(x) ** p for x in X.data) ** (1. / p)
        m_one = lambda X: abs(X).sum(axis=0).min() if 1 not in X.shape else sum(
            abs(x) ** p for x in X.data) ** (1. / p)
        v = {
            "fro": fro,
            "inf": inf,
            "-inf": m_inf,
            1: one,
            -1: m_one,
        }.get(p)
        return v(X) if v != None else sum(abs(x) ** p for x in X.data) ** (1. / p)
    else:
        return nla.norm(np.mat(X), p)


def vstack(X, format=None, dtype=None):
    """
    Stack sparse or dense matrices vertically (row wise).

    :param X: Sequence of matrices with compatible shapes.
    :type X: sequence of :class:`scipy.sparse` of format csr, csc, coo, bsr,
    dok, lil, dia or :class:`numpy.matrix`
    """
    if len([0 for x in X if not sp.isspmatrix(x)]) == 0:
        # scipy.sparse bug
        # return sp.vstack(X, format = X[0].getformat() if format is None else
        # format, dtype = X[0].dtype if dtype is None else dtype)
        return sp.vstack(X)
    else:
        return np.vstack(X)


def hstack(X, format=None, dtype=None):
    """
    Stack sparse or dense matrices horizontally (column wise).

    :param X: Sequence of matrices with compatible shapes.
    :type X: sequence of :class:`scipy.sparse` of format csr, csc, coo, bsr,
    dok, lil, dia or :class:`numpy.matrix`
    """
    if len([0 for x in X if not sp.isspmatrix(x)]) == 0:
        # scipy.sparse bug
        # return sp.hstack(X, format = X[0].getformat() if format is None else
        # format, dtype = X[0].dtyoe if dtype is None else dtype)
        return sp.hstack(X)
    else:
        return np.hstack(X)


def max(X, s):
    """
    Compute element-wise max(x,s) assignment for sparse or dense matrix.

    :param X: The input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    :param s: the input scalar.
    :type s: `float`
    """
    if sp.isspmatrix(X):
        Y = X.tocsr()
        DD = Y.data.copy()
        DD = np.maximum(DD, s)
        return sp.csr_matrix((DD, Y.indices, Y.indptr), Y.shape)
    else:
        return np.maximum(X, s)


def min(X, s):
    """
    Compute element-wise min(x,s) assignment for sparse or dense matrix.

    :param X: The input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    :param s: the input scalar.
    :type s: `float`
    """
    if sp.isspmatrix(X):
        Y = X.tocsr()
        DD = Y.data.copy()
        DD = np.minimum(DD, s)
        return sp.csr_matrix((DD, Y.indices, Y.indptr), Y.shape)
    else:
        return np.minimum(X, s)


def count(X, s):
    """
    Return the number of occurrences of element :param:`s` in sparse or
    dense matrix X.

    :param X: The input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    :param s: the input scalar.
    :type s: `float`
    """
    if sp.isspmatrix(X):
        return sum([1 for x in X.data if s == x])
    else:
        return sum([1 for r in X.tolist() for x in r if s == x])


def nz_data(X):
    """
    Return list of nonzero elements from X (! data, not indices).

    :param X: The input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    or :class:`numpy.matrix`
    """
    if sp.isspmatrix(X):
        return X.data.tolist()
    else:
        return [x for r in X.tolist() for x in r if x != 0]


def choose(n, k):
    """
    A fast way to calculate binomial coefficients C(n, k). It is 10 times faster
    than scipy.mis.comb for exact answers.

    :param n: Index of binomial coefficient.
    :type n: `int`
    :param k: Index of binomial coefficient.
    :type k: `int`
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0
