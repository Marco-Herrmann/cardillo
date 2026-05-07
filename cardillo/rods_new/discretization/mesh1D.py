import numpy as np
from numpy.polynomial import Polynomial
from scipy.sparse import lil_array
from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey
from .interpolations import lagrange, hermite

from cardillo.rods.discretization.gauss import gauss, lobatto, trapezoidal


class Mesh1D_equidistant:
    def __init__(
        self,
        basis,
        nelement,
        polynomial_degere,
        derivative_order,
    ):
        assert basis in ["Lagrange", "Lagrange_Disc", "Hermite_C0", "Hermite_C1"]

        self.basis = basis
        self.nelement = nelement
        self.polynomial_degere = polynomial_degere
        self.derivative_order = derivative_order

        # element boundaries
        self.xis_element = np.linspace(0, 1, self.nelement + 1)
        self.element_interval = np.array(
            [self.xis_element[:-1], self.xis_element[1:]]
        ).T

        # number of nodes
        if basis == "Lagrange":
            self.nnodes = self.npolynoms = polynomial_degere * nelement + 1
            self.nnodes_element = self.npolynoms_element = polynomial_degere + 1
            self.col = (
                lambda el: np.arange(self.npolynoms_element) + el * polynomial_degere
            )
            polynom_generator = lagrange

        elif basis == "Lagrange_Disc":
            self.nnodes = self.npolynoms = (polynomial_degere + 1) * nelement
            self.nnodes_element = self.npolynoms_element = polynomial_degere + 1
            self.col = lambda el: np.arange(self.npolynoms_element) + el * (
                polynomial_degere + 1
            )
            polynom_generator = lagrange

        elif basis == "Hermite_C0":
            assert polynomial_degere % 2 == 1, "polynomial_degere must be odd"
            self.nnodes = (polynomial_degere - 1) // 2 * nelement + 1
            self.nnodes_element = (polynomial_degere + 1) // 2
            self.npolynoms = polynomial_degere * nelement + 1
            self.npolynoms_element = polynomial_degere + 1
            polynom_generator = hermite

            def col(el):
                # ordering is (rs, ts^+, ts, ts^-)
                DOF_r = np.arange(self.nnodes_element) + el * (self.nnodes_element - 1)
                DOF_t_plus = self.nnodes + el
                DOF_t_middle = (
                    self.nnodes
                    + self.nelement
                    + np.arange(self.nnodes_element - 2)
                    + el * (self.nnodes_element - 2)
                )
                DOF_t_minus = (
                    self.nnodes + self.nelement * (self.nnodes_element - 1) + el
                )
                return np.array([*DOF_r, DOF_t_plus, *DOF_t_middle, DOF_t_minus])

            self.col = col

        elif basis == "Hermite_C1":
            raise NotImplementedError
            # assert polynomial_degere % 2 == 1, "polynomial_degere must be odd"
            # self.nnodes = (polynomial_degere - 1) // 2 * nelement + 1
            # self.nnodes_element = (polynomial_degere + 1) // 2
            # self.npolynoms = 2 * self.nnodes
            # self.npolynoms_element = 2 * self.nnodes_element
            # self.offset = polynomial_degere
            polynom_generator = hermite

        # xis in element
        xis_nodes_element = np.array(
            [
                np.linspace(
                    self.xis_element[el], self.xis_element[el + 1], self.nnodes_element
                )
                for el in range(self.nelement)
            ]
        )

        if basis in ["Lagrange", "Hermite_C0", "Hermite_C1"]:
            self.xis_nodes = np.linspace(0, 1, self.nnodes)
        elif basis == "Lagrange_Disc":
            self.xis_nodes = np.concatenate(xis_nodes_element)

        # shape functions
        polynomials = np.empty(
            (self.derivative_order + 1, self.nelement, self.npolynoms_element),
            dtype=Polynomial,
        )

        for el in range(self.nelement):
            polynomials[:, el, :] = polynom_generator(
                xis_nodes_element[el], self.derivative_order
            )

        self.polynomials = polynomials

    # TODO: vectorize
    # def node_numer(self, xis):
    def node_number(self, xi):
        # TODO: check to be consistent with element number for lagrange disc!
        """For given xi in I = [0.0, 1.0], returns node number if xi is a node, otherwise False"""
        idx = np.where(self.xis_nodes == xi)[0]
        if len(idx) == 1:
            return idx[0]
        else:
            return False

    def element_number(self, xis):
        """returns the element number(s) for xi, such that xi_{element} <= xi < xi_{element + 1}, expect for the last element, where xi_{element} <= xi <= xi_{element + 1}"""
        xis = np.asarray(xis)
        was_scalar = xis.ndim == 0
        xis = np.atleast_1d(xis)

        element_numbers = np.searchsorted(self.xis_element[:-1], xis, side="right") - 1
        # also allows for out of bounds
        element_numbers = np.clip(element_numbers, 0, self.nelement - 1)
        return element_numbers[0] if was_scalar else element_numbers

    # TODO: naming of functions
    def shape_functions_element(self, xis, el, derivative_order):
        xis = np.asarray(xis)
        was_scalar = xis.ndim == 0
        xis = np.atleast_1d(xis)

        N_dense = np.zeros((derivative_order + 1, len(xis), self.npolynoms_element))
        for d in range(derivative_order + 1):
            for p in range(self.npolynoms_element):
                N_dense[d, :, p] = self.polynomials[d, el, p](xis)

        return N_dense[:, 0, :] if was_scalar else N_dense

    def shape_functions_element_global(self, xis, el, derivative_order):
        N_sparse = []
        Nd = self.shape_functions_element(xis, el, derivative_order)
        col = self.col(el)
        for d in range(derivative_order + 1):
            Nd_sparse = lil_array((len(xis), self.npolynoms))
            Nd_sparse[:, col] = Nd[d]
            N_sparse.append(Nd_sparse)

        return N_sparse

    def shape_functions(self, xis, els=None, derivative_order=0):
        xis = np.atleast_1d(xis)
        nxis = len(xis)
        if els is None:
            els = self.element_number(xis)

        els = np.atleast_1d(els)
        nels = len(els)

        if nels != nxis:
            assert nels == 1, "Missmatch in lengths of given xi values and elements!"
            els = np.tile(els, nxis)

        N_sparse = [
            lil_array((len(xis), self.npolynoms)) for _ in range(derivative_order + 1)
        ]
        for el in np.unique(els):
            selection = els == el
            N_sparse_el = self.shape_functions_element_global(
                xis[selection], el, derivative_order
            )
            for d in range(derivative_order + 1):
                N_sparse[d][selection] = N_sparse_el[d]

        for d in range(derivative_order + 1):
            N_sparse[d] = N_sparse[d].tocsr()

        return N_sparse

    def quadrature(self, nquadrature, quadrature, derivative_order):
        nquadrature_total = self.nelement * nquadrature
        qp = np.empty(nquadrature_total, dtype=float)
        qw = np.empty(nquadrature_total, dtype=float)
        els = np.empty(nquadrature_total, dtype=int)

        if quadrature == "Gauss":
            quadrature_fct = gauss
        elif quadrature == "Lobatto":
            quadrature_fct = lobatto
        elif quadrature == "Trapezoidal":
            quadrature_fct = trapezoidal
        else:
            raise ValueError(f"Unknown quadrature {quadrature}")

        for el in range(self.nelement):
            idx = slice(el * nquadrature, (el + 1) * nquadrature)
            qp[idx], qw[idx] = quadrature_fct(
                nquadrature, interval=self.element_interval[el]
            )
            els[idx] = el

        N = self.shape_functions(qp, els, derivative_order)
        return dict(
            nquadrature_total=nquadrature_total,
            qp=qp,
            qw=qw,
            els=els,
            N=N,
        )

    def shape_function_array_element(self, xi, el, derivative):
        return np.array(
            [
                self.polynomials[derivative, el, p](xi)
                for p in range(self.npolynoms_element)
            ]
        )
