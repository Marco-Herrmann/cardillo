import numpy as np
from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey
from .lagrange import LagrangeBasis
from .gauss import gauss, lobatto


class Mesh1D:
    def __init__(
        self,
        knot_vector,
        nquadrature,
        dim_q,
        derivative_order=1,
        basis="Lagrange",
        quadrature="Gauss",
        dim_u=None,
    ):
        self.basis = basis
        self.nelement = knot_vector.nel
        self.knot_vector = knot_vector
        self.data = self.knot_vector.data
        self.degree = self.knot_vector.degree
        self.derivative_order = derivative_order
        self.nquadrature = nquadrature
        if quadrature == "Gauss":
            self.quadrature = gauss
        elif quadrature == "Lobatto":
            self.quadrature = lobatto
        else:
            raise NotImplementedError(
                f"Quadrature method '{quadrature}' is not implemented!"
            )

        self.lagrangebasis = LagrangeBasis(self.degree)
        self._eval_basis_cache = LRUCache(self.nelement * self.degree + 1)

        # we might have different meshes for q and u, e.g. quaternions for
        # describing spatial rotations
        if dim_u is None:
            self.dim_u = dim_u = dim_q

        if basis in ["Lagrange", "Lagrange_Disc"]:
            self.nnodes_per_element = (
                self.degree + 1
            )  # number of nodes influencing each element
            self.dim_q = dim_q  # number of degrees of freedom per node
            self.dim_u = dim_u
            self.nbasis_element = (
                self.nnodes_per_element
            )  # number of basis function per element
        else:
            raise NotImplementedError("")
        self.nq_per_element = (
            self.nnodes_per_element * dim_q
        )  # total number of generalized coordinates per element
        self.nu_per_element = (
            self.nnodes_per_element * dim_u
        )  # total number of generalized velocities per element

        # Boolean connectivity matrix for element polynomial_degrees of
        # freedom. This is used to extract the element degrees of freedom via
        # q[elDOF[el]] = q_e = C^e * q.
        self.elDOF = np.zeros((self.nelement, self.nq_per_element), dtype=int)
        self.elDOF_u = np.zeros((self.nelement, self.nu_per_element), dtype=int)

        if basis == "Lagrange":
            # total number of nodes
            self.nnodes = self.degree * self.nelement + 1

            elDOF_el = np.concatenate(
                [
                    np.arange(self.nnodes_per_element) + i * self.nnodes
                    for i in range(dim_q)
                ]
            )
            elDOF_el_u = np.concatenate(
                [
                    np.arange(self.nnodes_per_element) + i * self.nnodes
                    for i in range(dim_u)
                ]
            )

            for el in range(self.nelement):
                self.elDOF[el] = elDOF_el + el * self.degree
                self.elDOF_u[el] = elDOF_el_u + el * self.degree

            self.vtk_cell_type = "VTB_LAGRANGE_CURVE"
        elif basis == "Lagrange_Disc":
            # total number of nodes
            self.nnodes = (self.degree + 1) * self.nelement

            elDOF_el = np.concatenate(
                [
                    np.arange(self.nnodes_per_element) + i * self.nnodes
                    for i in range(dim_q)
                ]
            )

            elDOF_el_u = np.concatenate(
                [
                    np.arange(self.nnodes_per_element) + i * self.nnodes
                    for i in range(dim_u)
                ]
            )

            for el in range(self.nelement):
                self.elDOF[el] = elDOF_el + el * (self.degree + 1)
                self.elDOF_u[el] = elDOF_el_u + el * (self.degree + 1)

            # TODO: check if VTK export works
            self.vtk_cell_type = "VTB_LAGRANGE_CURVE"

        # todal number of degrees of freedoms

        self.nq = self.nnodes * dim_q
        self.nu = self.nnodes * dim_u

        # construct the Boolean selection matrix that choses the coordinates
        # of an individual node via q[nodalDOF[a]] = C^a * q
        self.nodalDOF = np.arange(self.nq).reshape(self.nnodes, dim_q, order="F")
        self.nodalDOF_u = np.arange(self.nu).reshape(self.nnodes, dim_u, order="F")

        # Boolean connectivity matrix for nodal polynomial_degrees of freedom
        # inside each element. This is only required if multiple fields are
        # discretized. It is used as qe[nodalDOF_element_[a]] = q_e^a = C^a * q_e
        self.nodalDOF_element = np.arange(self.nq_per_element).reshape(
            self.nnodes_per_element, dim_q, order="F"
        )
        self.nodalDOF_element_u = np.arange(self.nu_per_element).reshape(
            self.nnodes_per_element, dim_u, order="F"
        )

        # transform quadrature points on element intervals
        self.qp, self.wp = self.quadrature_points(nquadrature)

        # evaluate element shape functions at quadrature points
        shape_functions = self.shape_functions(nquadrature, derivative_order)
        self.N = shape_functions[0]
        if derivative_order >= 1:
            self.N_xi = shape_functions[1]
            if derivative_order >= 2:
                self.N_xixi = shape_functions[2]

    def basis1D(self, xis, el, derivative_order):
        if self.basis == "Lagrange":
            return self.lagrange_basis1D(
                xis,
                el,
                derivative_order,
                squeeze=False,
            )
        elif self.basis == "Lagrange_Disc":
            return self.lagrange_basis1D(
                xis,
                el,
                derivative_order,
                squeeze=False,
            )

    def lagrange_basis1D(self, xis, els, derivative_order=None, squeeze=True):
        derivative_order = derivative_order or self.derivative_order
        xis = np.atleast_1d(xis)
        els = np.atleast_1d(els)
        nxis = len(xis)
        nels = len(els)

        if nels != nxis:
            assert nels == 1, "Missmatch in lengths of given xi values and elements!"
            els = np.tile(els, nxis)

        N = np.zeros((derivative_order + 1, nxis, self.degree + 1))
        for i, (xi, eli) in enumerate(zip(xis, els)):
            eli = self.knot_vector.element_number(xi)[0] if eli is None else eli
            interval = self.knot_vector.element_interval(eli)
            self.lagrangebasis.set_interval(interval)
            N[0, i] = self.lagrangebasis(xi)
            if derivative_order:
                for j in range(1, derivative_order + 1):
                    N[j, i] = self.lagrangebasis.deriv(xi, n=j)
        if squeeze:
            return N.squeeze()
        else:
            return N

    @cachedmethod(
        lambda self: self._eval_basis_cache,
        key=lambda self, xi, el=None, derivative_order=None: hashkey(
            xi, el, derivative_order
        ),
    )
    def eval_basis(self, xi, el=None, derivative_order=None):
        if self.basis == "Lagrange":
            return self.lagrange_basis1D(xi, el, derivative_order, squeeze=True)
        elif self.basis == "Lagrange_Disc":
            return self.lagrange_basis1D(xi, el, derivative_order, squeeze=False)

    def eval_basis_matrix_q(self, xi, el, derivative_order):
        return self._eval_basis_matrix(xi, el, derivative_order)[0]

    def eval_basis_matrix_u(self, xi, el, derivative_order):
        return self._eval_basis_matrix(xi, el, derivative_order)[1]

    # TODO: cache this
    def _eval_basis_matrix(self, xi, el, derivative_order):
        interval = self.knot_vector.element_interval(el)
        self.lagrangebasis.set_interval(interval)

        # TODO: make switching between q and u more easy!
        N_q = np.zeros((derivative_order + 1, self.dim_q, self.nq_per_element))
        N_u = np.zeros((derivative_order + 1, self.dim_u, self.nu_per_element))
        Ni = self.lagrangebasis(xi)

        eye_q = np.eye(self.dim_q, dtype=float)
        eye_u = np.eye(self.dim_u, dtype=float)
        for node in range(self.nnodes_per_element):
            qDOF = self.nodalDOF_element[node]
            uDOF = self.nodalDOF_element_u[node]
            N_q[0, :, qDOF] = eye_q * Ni[0, node]
            N_u[0, :, uDOF] = eye_u * Ni[0, node]

        # TODO: optimize this!
        for j in range(1, derivative_order + 1):
            Ni_deriv = self.lagrangebasis.deriv(xi, n=j)
            for node in range(self.nnodes_per_element):
                qDOF = self.nodalDOF_element[node]
                uDOF = self.nodalDOF_element_u[node]
                N_q[j, :, qDOF] = eye_q * Ni_deriv[0, node]
                N_u[j, :, uDOF] = eye_u * Ni_deriv[0, node]

        if derivative_order == 0:
            return N_q[0], N_u[0]
        return N_q, N_u

    def quadrature_points(self, nquadrature):
        qp = np.zeros((self.nelement, nquadrature))
        wp = np.zeros((self.nelement, nquadrature))

        for el in range(self.nelement):
            Xi_element_interval = self.knot_vector.element_interval(el)
            qp[el], wp[el] = self.quadrature(nquadrature, interval=Xi_element_interval)

        return qp, wp

    def shape_functions(self, nquadrature, derivative_order):
        qp, _ = self.quadrature_points(nquadrature)

        N = [np.zeros((self.nelement, nquadrature, self.nbasis_element))]
        if derivative_order > 0:
            N.append(np.zeros((self.nelement, nquadrature, self.nbasis_element)))
            if derivative_order > 1:
                N.append(np.zeros((self.nelement, nquadrature, self.nbasis_element)))

        for el in range(self.nelement):
            NN = self.basis1D(qp[el], el, derivative_order)
            # expression = "self.N"
            # for i in range(self.derivative_order):
            #     expression += ", self.N_" + (i + 1) * "xi"
            # eval(expression) = NN
            N[0][el] = NN[0]
            if derivative_order > 0:
                N[1][el] = NN[1]
                if derivative_order > 1:
                    N[2][el] = NN[2]

        return N

    def shape_functions_matrix(self, nquadrature, derivative_order):
        shape_q = (self.nelement, nquadrature, self.dim_q, self.nq_per_element)
        shape_u = (self.nelement, nquadrature, self.dim_u, self.nu_per_element)

        Nq = [np.zeros(shape_q, dtype=float) for _ in range(derivative_order + 1)]
        Nu = [np.zeros(shape_u, dtype=float) for _ in range(derivative_order + 1)]

        shape_functions = self.shape_functions(nquadrature, derivative_order)

        eye_q = np.eye(self.dim_q, dtype=float)
        eye_u = np.eye(self.dim_u, dtype=float)
        for el in range(self.nelement):
            for i in range(nquadrature):
                for node in range(self.nnodes_per_element):
                    qDOF = self.nodalDOF_element[node]
                    uDOF = self.nodalDOF_element_u[node]

                    for d in range(derivative_order + 1):
                        Nq[d][el, i, :, qDOF] = shape_functions[d][el, i, node] * eye_q
                        Nu[d][el, i, :, uDOF] = shape_functions[d][el, i, node] * eye_u

        return Nq, Nu


if __name__ == "__main__":
    print("hello world")
