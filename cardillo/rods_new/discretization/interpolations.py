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


def B_spline_basis(nel, p_tot, cont, deriv_order):
    # knot vector
    mult = p_tot - cont
    U = [0.0] * (p_tot + 1)
    for i in range(1, nel):
        U += [i / nel] * mult
    U += [1.0] * (p_tot + 1)
    U = np.array(U)

    def B_rec(el, i, p_eval):
        if p_eval == 0:
            if (p_tot + 1) + mult * el == i + 1:
                return Polynomial([1.0])
            return Polynomial([0.0])

        denom1 = U[i + p_eval] - U[i]
        denom2 = U[i + p_eval + 1] - U[i + 1]
        if denom1 == 0:
            left = Polynomial([0.0])
        else:
            lin_left = Polynomial([-U[i] / denom1, 1.0 / denom1])
            left = lin_left * B_rec(el, i, p_eval - 1)

        if denom2 == 0:
            right = Polynomial([0.0])
        else:
            lin_right = Polynomial([U[i + p_eval + 1] / denom2, -1.0 / denom2])
            right = lin_right * B_rec(el, i + 1, p_eval - 1)

        return left + right

    # number of nodes (total)
    N = len(U) - p_tot - 1

    polynomials = np.empty((deriv_order + 1, N, nel), dtype=Polynomial)
    for el in range(nel):
        for i in range(N):
            p_i_el = B_rec(el, i, p_tot)
            polynomials[0, i, el] = p_i_el

            for d in range(1, deriv_order + 1):
                polynomials[d, i, el] = p_i_el.deriv(d)

    return polynomials
