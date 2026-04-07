import numpy as np
from scipy.sparse import bsr_array


class SparseArrayBlocks:
    def __init__(self, shape, blocksize):
        self.shape = shape
        self.blocksize = blocksize

        self.block_dicts = []

    def create_block_dict(self, Na, Nb, weights=None):
        block_dict = {}  # key = (row_block, col_block), value = list of (i, value)
        assert Na.shape[0] == Nb.shape[0]
        neval = Na.shape[0]
        if weights is None:
            weights = np.ones(neval)
        for i in range(neval):
            Ni_outer = (Na[i][:, None] @ Nb[i][None, :]).tocoo()
            for r, c, N in zip(Ni_outer.row, Ni_outer.col, Ni_outer.data):
                block_dict.setdefault((r, c), []).append((i, N * weights[i]))

        block_positions = list(block_dict.keys())
        nblocks = len(block_positions)
        block_rows, block_cols = np.array(block_positions).T

        weights_matrix = np.zeros((nblocks, neval))
        for b, pos in enumerate(block_positions):
            for i, N in block_dict[pos]:
                weights_matrix[b, i] = N

        # TODO: check if we can avoid ordering here by ordering above the N/N_xi
        order = np.lexsort((block_cols, block_rows))
        # assert (order == np.arange(len(order))).all()
        if not (order == np.arange(len(order))).all():
            print("Reordered!")
            block_rows = block_rows[order]
            block_cols = block_cols[order]
            # blocks = blocks[order]

        # TODO: can we find an explicit expression for indptr?
        # TODO: check if it is Na.shape[1] + 1 or Nb.shape[1] + 1
        indptr = np.zeros(Na.shape[1] + 1, dtype=int)
        np.add.at(indptr, block_rows + 1, 1)
        indptr = np.cumsum(indptr)

        # TODO: indptr, cols, ... only once, avoid ordering!
        self.block_dicts.append(
            dict(
                weights_matrix=weights_matrix,
                block_cols=block_cols,
                indptr=indptr,
                order=order,
            )
        )

    # TODO: create function to combine block dicts
    def add_blocks(self, qp_contributions):
        # TODO: improve this function
        for i, (qp_contr, bd) in enumerate(zip(qp_contributions, self.block_dicts)):
            weights_matrix = bd["weights_matrix"]
            block_cols = bd["block_cols"]
            indptr = bd["indptr"]
            order = bd["order"]

            # TODO: can this be done by matmul? or one einsum for all?
            blocks = np.einsum(
                "bi,ikl->bkl",
                weights_matrix,  # [iBlock, qpi]
                qp_contr,  # [qpi, rowDOF, colDOF]
            )

            # TODO: check if we can avoid ordering here by ordering above
            if i == 0:
                result = bsr_array(
                    (blocks[order], block_cols, indptr),
                    # (blocks, block_cols, indptr),
                    shape=self.shape,
                    blocksize=self.blocksize,
                )
            else:
                result += bsr_array(
                    (blocks[order], block_cols, indptr),
                    # (blocks, block_cols, indptr),
                    shape=self.shape,
                    blocksize=self.blocksize,
                )

        return result
