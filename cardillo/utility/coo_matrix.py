import warnings
from scipy.sparse import csc_array, csr_array, coo_array
from scipy.sparse._sputils import isshape, check_shape
from scipy.sparse import spmatrix, sparray
from numpy import repeat, tile, atleast_1d, atleast_2d, arange
from array import array


class CooMatrix:
    """Small container storing the sparse matrix shape and three lists for
    accumulating the entries for row, column and data Wiki/COO.

    Parameters
    ----------
    shape : tuple, 2D
        tuple defining the shape of the matrix

    References
    ----------
    Wiki/COO: https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)
    """

    def __init__(self, shape, nallocate=None):
        # check shape input
        if isinstance(shape, tuple):
            pass
        else:
            try:
                shape = tuple(shape)
            except Exception:
                raise ValueError(
                    "input argument shape is not tuple or cannot be interpreted as tuple"
                )

        # see https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/sputils.py#L210
        if isshape(shape, nonneg=True):
            M, N = shape
            # see https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/sputils.py#L267
            self.shape = check_shape((M, N))
        else:
            raise TypeError(
                "input argument shape cannot be interpreted as correct shape"
            )

        # python array as efficient container for numerical data,
        # see https://docs.python.org/3/library/array.html
        self.__data = array("d", [])  # double
        self.__row = array("I", [])  # unsigned int
        self.__col = array("I", [])  # unsigned int

        self.nallocate = nallocate
        self.allocated = nallocate is not None
        if self.allocated:
            self.fixed_size = False
            self.__idx0 = 0
            self.allocation_slices = [None] * nallocate

            self.allocated_sparse_array = [None] * nallocate

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, value):
        self.__data = value

    @property
    def row(self):
        return self.__row

    @row.setter
    def row(self, value):
        self.__row = value

    @property
    def col(self):
        return self.__col

    @col.setter
    def col(self, value):
        self.__col = value

    def __setitem__(self, key, value):
        if self.allocated:
            self.__setitem_allocated(key, value)
        else:
            self.__setitem_default(key[-2:], value)

    def __setitem_default(self, key, value):
        # None is returned by every function that does not return. Hence, we
        # can use this to add no contribution to the matrix.
        if value is not None:
            # extract rows and columns to assign
            rows, cols = key
            if isinstance(rows, slice):
                rows = arange(*rows.indices(self.shape[0]))
            if isinstance(cols, slice):
                cols = arange(*cols.indices(self.shape[1]))
            rows = atleast_1d(rows)
            cols = atleast_1d(cols)

            if isinstance(value, CooMatrix):
                assert value.shape == (len(rows), len(cols)), "inconsistent assignment"

                # extend arrays from given CooMatrix
                self.data.extend(value.data)
                self.row.extend(rows[value.row])
                self.col.extend(cols[value.col])
                # TODO: benchmark
                # self.data.fromlist(value.data.tolist())
                # self.row.fromlist(rows[value.row].tolist())
                # self.col.fromlist(cols[value.col].tolist())
            elif isinstance(value, sparray):
                assert value.shape == (len(rows), len(cols)), "inconsistent assignment"

                # all scipy sparse matrices are converted to coo_array, their
                # data, row and column lists are subsequently appended
                coo = value.tocoo()
                self.data.extend(coo.data)
                self.row.extend(rows[coo.row])
                self.col.extend(cols[coo.col])
                # TODO: benchmark
                # self.data.fromlist(coo.data.tolist())
                # self.row.fromlist(rows[coo.row].tolist())
                # self.col.fromlist(cols[coo.col].tolist())
            elif isinstance(value, spmatrix):
                raise RuntimeError("Do not use sparse matrices, move to sparse array.")
            else:
                # convert everything als to 2D numpy arrays
                value = atleast_2d(value)
                assert value.shape == (len(rows), len(cols)), "inconsistent assignment"

                # 2D array
                self.data.extend(value.ravel(order="C"))
                self.row.extend(repeat(rows, len(cols)))
                self.col.extend(tile(cols, len(rows)))

    def __setitem_allocated(self, key, value):
        if not self.fixed_size:
            # fill rows, columns and data, and save indices
            if value is not None:
                allocate_id, rows, cols = key
                if isinstance(rows, slice):
                    rows = arange(*rows.indices(self.shape[0]))
                if isinstance(cols, slice):
                    cols = arange(*cols.indices(self.shape[1]))
                rows = atleast_1d(rows)
                cols = atleast_1d(cols)

                if isinstance(value, CooMatrix):
                    assert value.shape == (
                        len(rows),
                        len(cols),
                    ), "inconsistent assignment"

                    # extend arrays from given CooMatrix
                    self.data.extend(value.data)
                    self.row.extend(rows[value.row])
                    self.col.extend(cols[value.col])
                    # TODO: benchmark
                    # self.data.fromlist(value.data.tolist())
                    # self.row.fromlist(rows[value.row].tolist())
                    # self.col.fromlist(cols[value.col].tolist())

                    self.allocation_slices[allocate_id] = slice(
                        self.__idx0, self.__idx0 + len(value.data)
                    )
                    self.__idx0 += len(value.data)

                elif isinstance(value, sparray):
                    assert value.shape == (
                        len(rows),
                        len(cols),
                    ), "inconsistent assignment"

                    # all scipy sparse matrices are converted to coo_array, their
                    # data, row and column lists are subsequently appended
                    coo = value.tocoo()
                    self.data.extend(coo.data)
                    self.row.extend(rows[coo.row])
                    self.col.extend(cols[coo.col])

                    self.allocation_slices[allocate_id] = slice(
                        self.__idx0, self.__idx0 + len(coo.data)
                    )
                    self.__idx0 += len(coo.data)

                    # TODO: check if row and col did not change if so: don't change anything but raise error
                    # self.allocated_sparse_array[]

                elif isinstance(value, spmatrix):
                    raise RuntimeError(
                        "Do not use sparse matrices, move to sparse array."
                    )
                else:
                    value = atleast_2d(value)
                    assert value.shape == (
                        len(rows),
                        len(cols),
                    ), "inconsistent assignment"

                    # 2D array
                    self.data.extend(value.ravel(order="C"))
                    self.row.extend(repeat(rows, len(cols)))
                    self.col.extend(tile(cols, len(rows)))

                    self.allocation_slices[allocate_id] = slice(
                        self.__idx0, self.__idx0 + value.size
                    )
                    self.__idx0 += value.size

            if key[0] == self.nallocate - 1:
                self.fixed_size = True
            return

        if value is not None:
            if isinstance(value, CooMatrix):
                data = value.data
            elif isinstance(value, sparray):
                coo = value.tocoo()
                data = coo.data
                # TODO: check if row/col changed
            else:
                data = atleast_2d(value).ravel(order="C")
            self.data[self.allocation_slices[key[0]]] = array("d", data)

    def extend(self, matrix, DOF):
        warnings.warn(
            "Usage of `CooMatrix.extend` is deprecated. "
            "You can simply index the object, e.g., coo[rows, cols] = value",
            category=DeprecationWarning,
        )
        self[DOF[0], DOF[1]] = matrix

    def asformat(self, format, copy=False):
        """Return this matrix in the passed format.
        Parameters
        ----------
        format : {str, None}
            The desired matrix format ("csr", "csc", "lil", "dok", "array", ...)
            or None for no conversion.
        copy : bool, optional
            If True, the result is guaranteed to not share data with self.
        Returns
        -------
        A : This matrix in the passed format.
        """
        if format is None:
            assert not copy, "copy=True is not supported when format=None"
            return self

        try:
            convert_method = getattr(self, "to" + format)
        except AttributeError as e:
            raise ValueError("Format {} is unknown.".format(format)) from e

        # Forward the copy kwarg, if it's accepted.
        try:
            return convert_method(copy=copy)
        except TypeError:
            return convert_method()

    def tosparse(self, scipy_matrix, copy=False):
        """Convert container to scipy sparse matrix.

        Parameters
        ----------
        scipy_matrix: scipy.sparse.spmatrix
            scipy sparse matrix format that should be returned
        """
        return scipy_matrix(
            (self.data, (self.row, self.col)), shape=self.shape, copy=copy
        )

    def tocoo(self, copy=False):
        """Convert container to scipy coo_array."""
        return self.tosparse(coo_array, copy=copy)

    def tocsc(self, copy=False):
        """Convert container to scipy csc_array."""
        return self.tosparse(csc_array, copy=copy)

    def tocsr(self, copy=False):
        """Convert container to scipy csr_array."""
        return self.tosparse(csr_array, copy=copy)

    def toarray(self, copy=False):
        """Convert container to 2D numpy array."""
        return self.tocoo(copy).toarray()


if __name__ == "__main__":
    from profilehooks import profile
    import numpy as np
    from scipy.sparse import random

    entries = 1
    density = 1
    local_size = 10
    nlocal = 100

    @profile(entries=entries)
    def run_dense_matrix():
        global_size = nlocal * local_size
        coo = CooMatrix((global_size, global_size))

        for i in range(nlocal):
            dense = np.random.rand(local_size, local_size)
            idx = np.arange(i * local_size, (i + 1) * local_size)
            coo[idx, idx] = dense

        return coo.tocsr()

    @profile(entries=entries)
    def run_dense_vector():
        global_size = nlocal * local_size
        coo = CooMatrix((global_size, global_size))

        for i in range(nlocal):
            dense = np.random.rand(local_size)
            idx = np.arange(i * local_size, (i + 1) * local_size)
            coo[idx, idx] = dense

        return coo.tocsr()

    @profile(entries=entries)
    def run_scipy_sparse():
        global_size = nlocal * local_size
        coo = CooMatrix((global_size, global_size))

        for i in range(nlocal):
            dense = random(local_size, local_size, density=density)
            idx = np.arange(i * local_size, (i + 1) * local_size)
            coo[idx, idx] = dense

        return coo.tocsr()

    @profile(entries=entries)
    def run_coo_sparse():
        global_size = nlocal * local_size
        coo = CooMatrix((global_size, global_size))

        for i in range(nlocal):
            dense = random(local_size, local_size, density=density)
            dense_coo = CooMatrix((local_size, local_size))
            dense_coo.data = array("d", dense.data)
            dense_coo.row = array("I", dense.row)
            dense_coo.col = array("I", dense.col)
            idx = np.arange(i * local_size, (i + 1) * local_size)
            coo[idx, idx] = dense_coo

        return coo.tocsr()

    run_dense_matrix()
    run_dense_vector()
    run_scipy_sparse()
    run_coo_sparse()
