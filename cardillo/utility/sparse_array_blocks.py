import numpy as np
from scipy.sparse import bsr_array


class SparseArrayBlocks:
    def __init__(self, max_shape, max_blocksize, pairs, slices=None):
        self.max_shape = max_shape
        self.max_blocksize = max_blocksize
        if slices == None:
            self.create_bsr = lambda blocks: self.create_bsr_single(
                blocks, max_shape, max_blocksize
            )

        else:
            nR = max_shape[0] // max_blocksize[0]
            nC = max_shape[1] // max_blocksize[1]

            # 0 -> None, 1 -> full, 2 -> sliced
            # 00, 01, 02 -> 0, 11 -> 1, 12, 22 -> 2
            bsr_type = np.empty(len(slices), dtype=int)
            shapes = [max_shape] * len(slices)
            blocksizes = [max_blocksize] * len(slices)
            for i, slice_i in enumerate(slices):
                bsr_type_i = np.empty(2, dtype=int)
                n_i = [0, 0]
                for j, row_col in enumerate(slice_i):
                    if isinstance(row_col, slice):
                        if row_col.start == row_col.stop:
                            bsr_type_i[j] = 0
                        if (
                            row_col.start == 0
                            and row_col.stop == max_blocksize[j]
                            and row_col.step == 1
                        ):
                            bsr_type_i[j] = 1
                        else:
                            bsr_type_i[j] = 2
                        raise NotImplementedError
                        n_i[j] = ...  # TODO

                    elif type(row_col) is type(Ellipsis):
                        bsr_type_i[j] = 1
                        n_i[j] = max_blocksize[j]

                    elif isinstance(row_col, np.ndarray):
                        if len(row_col) == 0:
                            bsr_type_i[j] = 0
                        elif len(row_col) == max_blocksize[j] and np.all(
                            row_col == np.arange(max_blocksize[j])
                        ):
                            bsr_type_i[j] = 1
                        else:
                            bsr_type_i[j] = 2
                        n_i[j] = row_col.shape[0]

                if np.any(bsr_type_i == 0):
                    bsr_type[i] = 0
                elif np.any(bsr_type_i == 2):
                    bsr_type[i] = 2
                    blocksizes[i] = n_i
                    shapes[i] = (nR * n_i[0], nC * n_i[1])
                else:
                    bsr_type[i] = 1

            self.multiples = (bsr_type, shapes, blocksizes, slices)
            self.create_bsr = self.create_bsr_multiple
            self.nresults = len(slices)

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
        self.block_cols = block_positions[:, 1]
        counts = np.bincount(block_positions[:, 0], minlength=Na.shape[1])
        self.indptr = np.concatenate(([0], np.cumsum(counts)))

    def add_blocks(self, qp_contributions):
        # numpy equivalent to einsum ("pinm, pbi -> bnm", qp_contr, weights)
        # Note: reshape shares memory!
        tmp = self.weights_matrix @ qp_contributions.reshape(
            *qp_contributions.shape[:2], -1
        )
        blocks = tmp.sum(axis=0).reshape(-1, *qp_contributions.shape[2:])

        return self.create_bsr(blocks)

    def create_bsr_single(self, blocks, shape, blocksize):
        # Note: bsr like this keeps zeros
        result = bsr_array(
            (blocks, self.block_cols, self.indptr),
            shape=shape,
            blocksize=blocksize,
        )

        return result

    def create_bsr_multiple(self, blocks):
        result = [None] * self.nresults
        for i, (btype_i, shape_i, blocksize_i, slice_i) in enumerate(
            zip(*self.multiples)
        ):
            if btype_i == 0:
                result[i] = None
            elif btype_i == 1:
                result[i] = self.create_bsr_single(
                    blocks, self.max_shape, self.max_blocksize
                )
            elif btype_i == 2:
                result[i] = self.create_bsr_single(
                    blocks[:, *slice_i], shape_i, blocksize_i
                )

        return result
