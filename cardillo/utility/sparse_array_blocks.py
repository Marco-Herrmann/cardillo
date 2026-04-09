import numpy as np
from scipy.sparse import bsr_array


class SparseArrayBlocks:
    def __init__(self, shape, blocksize, pairs):
        self.shape = shape
        self.blocksize = blocksize

        nPairs = len(pairs)
        neval = pairs[0][0].shape[0]

        block_dict = {}  # key = (row_block, col_block), value = list of (i, value)
        for p, (Na, Nb, weights) in enumerate(pairs):
            assert neval == Na.shape[0], "Dimensions missmatch"
            assert neval == Nb.shape[0], "Dimensions missmatch"

            for i in range(neval):
                Ni_outer = (Na[i][:, None] @ Nb[i][None, :]).tocoo()
                for r, c, N in zip(Ni_outer.row, Ni_outer.col, Ni_outer.data):
                    block_dict.setdefault((r, c), []).append((p, i, N * weights[i]))

        block_positions = np.array(list(block_dict.keys()))
        block_rows, block_cols = block_positions.T

        order = np.lexsort((block_cols, block_rows))
        block_positions = block_positions[order]

        self.weights_matrix = np.zeros((nPairs, block_positions.shape[0], neval))
        for b, pos in enumerate(block_positions):
            for p, i, N in block_dict[*pos]:
                self.weights_matrix[p, b, i] = N

        self.block_cols = block_cols[order]
        # TODO: can we find an explicit expression for indptr?
        # TODO: check if it is Na.shape[1] + 1 or Nb.shape[1] + 1
        indptr = np.zeros(Na.shape[1] + 1, dtype=int)
        np.add.at(indptr, block_rows + 1, 1)
        self.indptr = np.cumsum(indptr)

    def add_blocks(self, qp_contributions):
        # numpy equivalent to einsum ("pinm, pbi -> bnm", qp_contr, weights)
        # Note: reshape shares memory!
        blocks = (
            (
                self.weights_matrix
                @ qp_contributions.reshape(*qp_contributions.shape[:2], -1)
            )
            .sum(axis=0)
            .reshape(-1, *qp_contributions.shape[2:])
        )

        # Note: bsr like this keeps zeros
        result = bsr_array(
            (blocks, self.block_cols, self.indptr),
            shape=self.shape,
            blocksize=self.blocksize,
        )

        return result
