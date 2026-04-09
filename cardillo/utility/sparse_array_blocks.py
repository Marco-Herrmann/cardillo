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
                Nai_col, Nai_data = Na[i].col, Na[i].data
                Nbi_col, Nbi_data = Nb[i].col, Nb[i].data
                wi = weights[i]

                # iterate rows via Na
                for rr, vr in zip(Nai_col, Nai_data):
                    # iterate cols via Nb
                    for cc, vc in zip(Nbi_col, Nbi_data):
                        block_dict.setdefault((rr, cc), []).append((p, i, vr * vc * wi))

        # get and sort block positions
        block_positions = np.array(list(block_dict.keys()))
        order = np.lexsort((block_positions[:, 1], block_positions[:, 0]))
        block_positions = block_positions[order]

        # fill weight matrix
        pos_to_idx = {tuple(pos): b for b, pos in enumerate(block_positions)}
        self.weights_matrix = np.zeros((nPairs, block_positions.shape[0], neval))
        for (r, c), entries in block_dict.items():
            b = pos_to_idx[(r, c)]
            for p, i, N in entries:
                self.weights_matrix[p, b, i] = N

        # prepare for bsr
        self.block_cols = block_positions[order, 1]
        counts = np.bincount(block_positions[order, 0], minlength=Na.shape[1])
        self.indptr = np.concatenate(([0], np.cumsum(counts)))

    def add_blocks(self, qp_contributions):
        # numpy equivalent to einsum ("pinm, pbi -> bnm", qp_contr, weights)
        # Note: reshape shares memory!
        tmp = self.weights_matrix @ qp_contributions.reshape(
            *qp_contributions.shape[:2], -1
        )
        blocks = tmp.sum(axis=0).reshape(-1, *qp_contributions.shape[2:])

        # Note: bsr like this keeps zeros
        result = bsr_array(
            (blocks, self.block_cols, self.indptr),
            shape=self.shape,
            blocksize=self.blocksize,
        )

        return result
