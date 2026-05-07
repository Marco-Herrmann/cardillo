import numpy as np
from numpy.polynomial import Polynomial


def lagrange(xis_nodes, deriv_order):
    nnodes = len(xis_nodes)
    polynomials = np.zeros((deriv_order + 1, nnodes), dtype=Polynomial)
    for node_i in range(nnodes):
        xi_node_i = xis_nodes[node_i]
        poly_i = Polynomial([1.0])

        for node_j in range(nnodes):
            if node_i != node_j:
                xi_node_j = xis_nodes[node_j]
                inv_diff = 1.0 / (xi_node_i - xi_node_j)
                poly_i *= Polynomial([-xi_node_j * inv_diff, inv_diff])

        for d in range(deriv_order + 1):
            polynomials[d, node_i] = poly_i.deriv(d)

    return polynomials


def hermite(xis_nodes, deriv_order):
    nnodes = len(xis_nodes)
    polynomials_l = lagrange(xis_nodes, 1)
    polynomials_h = np.zeros((deriv_order + 1, 2 * nnodes), dtype=Polynomial)

    for node_i in range(nnodes):
        xi_node_i = xis_nodes[node_i]
        pl2 = polynomials_l[0, node_i] ** 2
        pi = Polynomial([-xi_node_i, 1])
        h_i = pl2 * (1 - 2 * pi * polynomials_l[1, node_i](xi_node_i))
        h_Npi = pl2 * pi

        for d in range(deriv_order + 1):
            polynomials_h[d, node_i] = h_i.deriv(d)
            polynomials_h[d, nnodes + node_i] = h_Npi.deriv(d)

    return polynomials_h
