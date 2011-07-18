import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import numpy.linalg as nla
from operator import mul
from itertools import izip

"""
    Linear algebra helper routines
"""

#######
###    Wrapper functions for handling sparse matrices and dense matrices representation.
###    scipy.sparse, numpy.matrix
#######

def diff(X):
    """Compute differences between adjacent elements of X."""   
    assert 1 in X.shape, "X should be a vector."
    assert not sp.isspmatrix(X), "X is sparse matrix."
    X = X.flatten()
    return [X[0, j + 1] - X[0, j] for j in xrange(X.shape[1] - 1)]

def sub2ind(shape, row_sub, col_sub):
    """Return the linear index equivalents to the row and column subscripts for given matrix shape"""
    assert len(row_sub) == len(col_sub), "Row and column subscripts do not match."
    res = [j * shape[0] + i for i,j in zip(row_sub, col_sub)]
    return res

def any(X, axis = None):
    """Test whether any element along a given axis of sparse or dense matrix X are nonzero."""
    if sp.isspmatrix(X): 
        assert isinstance(X, sp.csr_matrix) or isinstance(X, sp.csc_matrix), "Incorrect sparse format."
        assert axis == 0 or axis == 1 or axis == None, "Incorrect axis number."
        if axis is None:
            return len(X.data) != X.shape[0] * X.shape[1]
        res = [0 for _ in xrange(X.shape[1 - axis])] 
        def _caxis(now, row, col):
            res[col] += 1
        def _raxis(now, row, col):
            res[row] += 1
        check = _caxis if axis == 0 else _raxis 
        now = 0
        for row in range(X.shape[0]):
            upto = X.indptr[row+1]
            while now < upto:
                col = X.indices[now]
                check(now, row, col)
                now += 1
        return [x != 0 for x in res]
    else:
        return X.any(axis)
        
def all(X, axis = None):
    """Test whether all elements along a given axis of sparse or dense matrix X are nonzero."""
    if sp.isspmatrix(X):
        assert isinstance(X, sp.csr_matrix) or isinstance(X, sp.csc_matrix), "Incorrect sparse format."
        assert axis == 0 or axis == 1 or axis == None, "Incorrect axis number."
        if axis is None:
            return len(X.data) == X.shape[0] * X.shape[1]
        res = [0 for _ in xrange(X.shape[1 - axis])] 
        def _caxis(now, row, col):
            res[col] += 1
        def _raxis(now, row, col):
            res[row] += 1
        check = _caxis if axis == 0 else _raxis
        now = 0
        for row in range(X.shape[0]):
            upto = X.indptr[row+1]
            while now < upto:
                col = X.indices[now]
                check(now, row, col)
                now += 1
        return [x == X.shape[0] if axis == 0 else x == X.shape[1] for x in res]
    else:
        return X.all(axis)

def find(X):
    """Return all nonzero elements indices (linear indices) of sparse or dense matrix X. It is Matlab notation."""
    if sp.isspmatrix(X):
        res = []
        now = 0
        for row in range(X.shape[0]):
            upto = X.indptr[row+1]
            while now < upto:
                col = X.indices[now]
                if X.data[now]:
                    res.append(col * X.shape[0] + row)
                now += 1
        return res
    else:
        return [j * X.shape[0] + i for i in xrange(X.shape[0]) for j in xrange(X.shape[1]) if X[i,j]]

def negative(X):
    """Check if X contains negative elements."""
    if sp.isspmatrix(X):
        if any(X.data < 0):
                return True
    else:
        if any(np.asmatrix(X) < 0):
            return True
        
def sort(X):
    """Return sorted elements of X and also array of indices."""
    assert 1 in X.shape, "X should be vector."
    X = X.flatten().tolist()[0]
    return sorted(X), sorted(range(len(X)), key = X.__getitem__)
    
    
def argmax(X, axis = None):
    """
    Return tuple of values and indices of the maximum entries along an axis. Row major order.
    :param X: sparse or dense matrix
    :type X: :class:`scipy.sparse.csr_matrix`, :class:`scipy.sparse.csc_matrix` or class:`numpy.matrix`
    :param axis: Specify axis along which to operate.
    :type axis: `int`
    """
    if sp.isspmatrix(X):
        assert isinstance(X, sp.csr_matrix) or isinstance(X, sp.csc_matrix), "Incorrect sparse format."
        assert axis == 0 or axis == 1 or axis == None, "Incorrect axis number."
        res = [[float('-inf'), 0] for _ in xrange(X.shape[1 - axis])] if axis is not None else [float('-inf'), 0]
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
        [check(row, col) for row in xrange(X.shape[0]) for col in xrange(X.shape[1])]
        if axis == None:
            return res
        elif axis == 0:
            t = zip(*res)
            return list(t[0]), np.matrix(t[1])
        else:
            t = zip(*res)
            return list(t[0]), np.matrix(t[1]).T
    else:
        idxX = np.asmatrix(X).argmax(axis)
        if axis == None:
            eX = X[idxX / X.shape[1], idxX % X.shape[1]]
        elif axis == 0:
            eX = [X[idxX[0,idx], col] for idx, col in izip(xrange(X.shape[1]), xrange(X.shape[1]))]
        else:
            eX = [X[row, idxX[idx, 0]] for row, idx in izip(xrange(X.shape[0]), xrange(X.shape[0]))]
        return eX, idxX 
    
def argmin(X, axis = None):
    """
    Return tuple of values and indices of the minimum entries along an axis. Row major order.
    :param X: sparse or dense matrix
    :type X: :class:`scipy.sparse.csr_matrix`, :class:`scipy.sparse.csc_matrix` or class:`numpy.matrix`
    :param axis: Specify axis along which to operate.
    :type axis: `int`
    """
    if sp.isspmatrix(X):
        assert isinstance(X, sp.csr_matrix) or isinstance(X, sp.csc_matrix), "Incorrect sparse format."
        assert axis == 0 or axis == 1 or axis == None, "Incorrect axis number."
        res = [[float('inf'), 0] for _ in xrange(X.shape[1 - axis])] if axis is not None else [float('inf'), 0]
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
        [check(row, col) for row in xrange(X.shape[0]) for col in xrange(X.shape[1])]
        if axis == None:
            return res
        elif axis == 0:
            t = zip(*res)
            return list(t[0]), np.matrix(t[1])
        else:
            t = zip(*res)
            return list(t[0]), np.matrix(t[1]).T
    else:
        idxX = np.asmatrix(X).argmin(axis)
        if axis == None:
            eX = X[idxX / X.shape[1], idxX % X.shape[1]]
        elif axis == 0:
            eX = [X[idxX[0,idx], col] for idx, col in izip(xrange(X.shape[1]), xrange(X.shape[1]))]
        else:
            eX = [X[row, idxX[idx, 0]] for row, idx in izip(xrange(X.shape[0]), xrange(X.shape[0]))]
        return eX, idxX 

def repmat(X, m, n):
    """Construct matrix consisting of an m-by-n tiling of copies of X."""
    if sp.isspmatrix(X):
        return sp.hstack([sp.vstack([X for _ in xrange(m)], format = X.format) for _ in xrange(n)], format = X.format)
    else:
        return np.tile(np.asmatrix(X), (m, n))
    
def svd(X, k):
    """Compute standard SVD on X."""
    if sp.isspmatrix(X): 
        U, S, V = sla.svd(X, k)
    else: 
        U, S, V = nla.svd(np.asmatrix(X))
    return U, S, V

def dot(X, Y):
    """Compute dot product of X and Y."""
    if sp.isspmatrix(X) and sp.isspmatrix(Y):
        return X * Y
    elif sp.isspmatrix(X) or sp.isspmatrix(Y):
        # avoid dense dot product with mixed factors
        # avoid copying sparse matrix
        return sp.csc_matrix(X) * sp.csr_matrix(Y)
    else:
        return np.asmatrix(X) * np.asmatrix(Y)

def multiply(X, Y):
    """Compute element-wise multiplication of X and Y."""
    if sp.isspmatrix(X) and sp.isspmatrix(Y):
        return X.multiply(Y)
    elif sp.isspmatrix(X) or sp.isspmatrix(Y):
        return _op_spmatrix(X, Y, mul) 
    else:
        return np.multiply(np.asmatrix(X), np.asmatrix(Y))
    
def sop(X, s = None, op = None):
    """Compute scalar element wise operation of matrix X and scalar."""
    if sp.isspmatrix(X):
        return _sop_spmatrix(X, s, op)
    else:
        return _sop_matrix(X, s, op)
    
def _sop_spmatrix(X, s = None, op = None):
    """Compute sparse scalar element wise operation of matrix X and scalar."""
    assert isinstance(X, sp.csr_matrix) or isinstance(X, sp.csc_matrix), "Incorrect sparse format."
    R = X.copy()
    now = 0
    for row in range(X.shape[0]):
        upto = X.indptr[row+1]
        while now < upto:
            R.data[now] = op(X.data[now], s) if s != None else op(X.data[now])
            now += 1
    return R

def _sop_matrix(X, s = None, op = None):
    """Compute scalar element wise operation of matrix X and scalar."""
    return np.matrix([[op(X[i,j], s) if s != None else op(X[i,j]) for j in xrange(X.shape[1])] for i in xrange(X.shape[0])])
    
def elop(X, Y, op):
    """Compute element-wise operation of matrix X and matrix Y."""
    try:
        zp = op(0, 1) if sp.isspmatrix(X) else op(1, 0)
    except:
        zp = 0
    if sp.isspmatrix(X) and sp.isspmatrix(Y):
        return _op_spmatrix(X, Y, op) if not zp else _op_matrix(X, Y, op)
    elif sp.isspmatrix(X) or sp.isspmatrix(Y):
        return _op_spmatrix(X, Y, op) if not zp else _op_matrix(X, Y, op)
    else:
        return op(np.asmatrix(X), np.asmatrix(Y))

def _op_spmatrix(X, Y, op):
    """Compute sparse element-wise operation for operations preserving zeros."""
    # distinction as op is not necessarily commutative
    return __op_spmatrixX(X, Y, op) if sp.isspmatrix(X) else __op_spmatrixY(X, Y, op)
    

def __op_spmatrixX(X, Y, op):
    """Compute sparse element-wise operation for operations preserving zeros."""
    assert isinstance(X, sp.csr_matrix) or isinstance(X, sp.csc_matrix), "Incorrect sparse format."
    assert X.shape == Y.shape, "Matrices are not aligned."
    R = X.copy()
    now = 0
    for row in range(X.shape[0]):
        upto = X.indptr[row+1]
        while now < upto:
            col = X.indices[now]
            R.data[now] = op(X.data[now], Y[row, col])
            now += 1
    return R
    
def __op_spmatrixY(X, Y, op):
    """Compute sparse element-wise operation for operations preserving zeros."""
    assert isinstance(Y, sp.csr_matrix) or isinstance(Y, sp.csc_matrix), "Incorrect sparse format."
    assert X.shape == Y.shape, "Matrices are not aligned."
    R = Y.copy()
    now = 0
    for row in range(Y.shape[0]):
        upto = Y.indptr[row+1]
        while now < upto:
            col = Y.indices[now]
            R.data[now] = op(X[row, col], Y.data[now])
            now += 1
    return R

def _op_matrix(X, Y, op):
    """Compute sparse element-wise operation for operations not preserving zeros."""
    # operation is not necessarily commutative 
    assert X.shape == Y.shape, "Matrices are not aligned."
    return np.matrix([[op(X[i,j], Y[i,j]) for j in xrange(X.shape[1])] for i in xrange(X.shape[0])])

def inf_norm(X):
    """Infinity norm of a matrix (maximum absolute row sum).
    :param X: sparse or dense matrix
    :type X: :class:`scipy.sparse.csr_matrix`, :class:`scipy.sparse.csc_matrix` or class:`numpy.matrix`
    """
    if sp.isspmatrix_csr(X) or sp.isspmatrix_csc(X):
        # avoid copying index and ptr arrays
        abs_X = X.__class__((abs(X.data), X.indices, X.indptr), shape = X.shape)
        return (abs_X * np.ones((X.shape[1]), dtype = X.dtype)).max()
    elif sp.isspmatrix(X):
        return (abs(X) * np.ones((X.shape[1]), dtype = X.dtype)).max()
    else:
        return nla.norm(np.asmatrix(X), float('inf'))

def norm(X, p = "fro"):
    """Compute entry-wise norms (! not induced/operator norms)."""
    assert 1 not in X.shape and p == 2, "Computing entrywise norms only."
    if sp.isspmatrix(X):
        v = {
         "fro": sum(abs(x)**2 for x in X.data)**(1. / 2),
         "inf": max(abs(X).sum(axis = 1)) if 1 not in X.shape else max(abs(X)),
        "-inf": min(abs(X).sum(axis = 1)) if 1 not in X.shape else min(abs(X)),
             1: max(abs(X).sum(axis = 0)) if 1 not in X.shape else sum(abs(x)**p for x in X.data)**(1. / p),
            -1: min(abs(X).sum(axis = 0)) if 1 not in X.shape else sum(abs(x)**p for x in X.data)**(1. / p)
            }
        return v.get(p, sum(abs(x)**p for x in X.data)**(1. / p))
    else:
        return nla.norm(np.asmatrix(X), p)
    
def vstack(X, format = None, dtype = None):
    """Stack sparse or dense matrices vertically."""
    if len([0 for x in X if not sp.isspmatrix(x)]) == 0:
        return sp.vstack(X, format, dtype)
    else:
        return np.vstack(X)

def hstack(X, format = None, dtype = None):
    """Stack sparse or dense matrices horizontally."""
    if len([0 for x in X if not sp.isspmatrix(x)]) == 0:
        return sp.hstack(X, format, dtype)
    else:
        return np.hstack(X)
    
def max(X, s):
    """Compute element-wise max(x,s) assignement for sparse or dense matrix."""
    if sp.isspmatrix(X):
        assert isinstance(X, sp.csr_matrix) or isinstance(X, sp.csc_matrix), "Incorrect sparse format."
        R = X.copy()
        now = 0
        for row in range(X.shape[0]):
            upto = X.indptr[row+1]
            while now < upto:
                col = X.indices[now]
                R.data[now] = max(X[row, col], s)
                now += 1
        return R
    else:
        return np.matrix([[max(X[i,j], s) for j in xrange(X.shape[1])] for i in xrange(X.shape[0])])
    
def min(X, s):
    """Compute element-wise min(x,s) assignement for sparse or dense matrix."""
    if sp.isspmatrix(X):
        assert isinstance(X, sp.csr_matrix) or isinstance(X, sp.csc_matrix), "Incorrect sparse format."
        R = X.copy()
        now = 0
        for row in range(X.shape[0]):
            upto = X.indptr[row+1]
            while now < upto:
                col = X.indices[now]
                R.data[now] = min(X[row, col], s)
                now += 1
        return R
    else:
        return np.matrix([[min(X[i,j], s) for j in xrange(X.shape[1])] for i in xrange(X.shape[0])])
    
def count(X, s):
    """Return the number of occurrences of element s in sparse or dense matrix X."""
    if sp.isspmatrix(X):
        return sum([1 for x in X.data if s == x])
    else:
        return sum([1 for r in X.tolist() for x in r if s == x])
    
def nz_data(X):
    """Return list of nonzero elements from X (! data, not indices)."""
    if sp.isspmatrix(X):
        return X.data.tolist()
    else:
        return [x for r in X.tolist() for x in r if x != 0]
    
    