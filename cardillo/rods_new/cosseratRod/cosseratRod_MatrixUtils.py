"""Helper functions for the (sparse) matrices of the rod finite elements.

The bsr functions keep the sparsity pattern of their input, including explicit
zeros. This is required, as System preallocates the pattern of the first call.
"""

import numpy as np
from scipy.sparse import bsr_array, coo_array

from cardillo.utility.coo_matrix import CooMatrix


##############
# bsr arrays #
##############
def bsr_block_rows(M):
    """Block row index of each block of the bsr array M."""
    return np.repeat(np.arange(len(M.indptr) - 1), np.diff(M.indptr))


def bsr_block_diag(blocks, shape=None):
    """Block diagonal bsr array of blocks (n, a, b), including zeros."""
    n, a, b = blocks.shape
    shape = (n * a, n * b) if shape is None else shape
    return bsr_array((blocks, np.arange(n), np.arange(n + 1)), shape=shape)


def bsr_left(L, M):
    """L @ M for block diagonal L with blocks L[i] (a, b) and M in bsr format with
    blocksize (b, c). The sparsity pattern of M is kept, including zeros."""
    data = np.einsum("nab,nbc->nac", L[bsr_block_rows(M)], M.data)
    return bsr_array(
        (data, M.indices, M.indptr), shape=(L.shape[0] * L.shape[1], M.shape[1])
    )


def bsr_right(M, R):
    """M @ R for block diagonal R with blocks R[j] (c, d) and M in bsr format with
    blocksize (b, c). The sparsity pattern of M is kept, including zeros."""
    data = np.einsum("nbc,ncd->nbd", M.data, R[M.indices])
    return bsr_array(
        (data, M.indices, M.indptr), shape=(M.shape[0], R.shape[0] * R.shape[2])
    )


def bsr_add_block_diag(M, D):
    """M + block_diag(D), where all diagonal blocks are part of the pattern of M."""
    rows = bsr_block_rows(M)
    diag = M.indices == rows
    data = M.data.copy()
    data[diag] += D[rows[diag]]
    return bsr_array((data, M.indices, M.indptr), shape=M.shape)


def bsr_sym_skew(M):
    """Symmetric and skew-symmetric part of a square bsr array with square blocks
    and symmetric block pattern. Both parts keep the sparsity pattern of M."""
    assert M.blocksize[0] == M.blocksize[1]
    rows = bsr_block_rows(M)
    position = {(r, c): k for k, (r, c) in enumerate(zip(rows, M.indices))}
    transposed = np.array([position[(c, r)] for r, c in zip(rows, M.indices)])
    M_T_data = M.data[transposed].transpose(0, 2, 1)
    sym = bsr_array((0.5 * (M.data + M_T_data), M.indices, M.indptr), shape=M.shape)
    skew = bsr_array((0.5 * (M.data - M_T_data), M.indices, M.indptr), shape=M.shape)
    return sym, skew


################
# dense arrays #
################
def dense_block_diag(blocks):
    """Block diagonal array of blocks (n, a, b) or blocks (n, a, b, c), where the
    latter results in an array (n * a, n * b, n * c) with blocks on the diagonal."""
    if blocks.ndim == 3:
        n, a, b = blocks.shape
        out = np.zeros((n, a, n, b), dtype=blocks.dtype)
        for i in range(n):
            out[i, :, i, :] = blocks[i]
        return out.reshape(n * a, n * b)

    n, a, b, c = blocks.shape
    out = np.zeros((n, a, n, b, n, c), dtype=blocks.dtype)
    for i in range(n):
        out[i, :, i, :, i, :] = blocks[i]
    return out.reshape(n * a, n * b, n * c)


################################
# composition of contributions #
################################
def to_sparse(A):
    """scipy sparse array of A (CooMatrix, sparse or dense array)."""
    return A.tocoo() if isinstance(A, CooMatrix) else A


def sum_sparse(matrices):
    """Sum of sparse arrays in coo format, where duplicate entries are not summed
    up, i.e., the sparsity pattern is kept, including zeros."""
    coos = [to_sparse(A).tocoo() for A in matrices]
    return coo_array(
        (
            np.concatenate([A.data for A in coos]),
            (
                np.concatenate([A.row for A in coos]),
                np.concatenate([A.col for A in coos]),
            ),
        ),
        shape=coos[0].shape,
    )


def sum_matrix_pairs(pairs, shape):
    """Sums pairs of matrices, e.g., [(K_1, N_1), (K_2, N_2), ...], where each entry
    can be None, a CooMatrix, a sparse or a dense array."""
    result = []
    for matrices in zip(*pairs):
        if all(matrix is None for matrix in matrices):
            result.append(None)
            continue
        coo = CooMatrix(shape)
        for matrix in matrices:
            coo[:, :] = matrix
        result.append(coo)
    return tuple(result)
