import numpy as np
import timeit

###############
# define Mesh #
###############
nelements = 1_000
nq_node = 7
p = 2

###############
# create mesh #
###############
nnodes = (p * nelements) + 1
nnodes_element = p + 1

nq = nq_node * nnodes
nq_element = nq_node * nnodes_element


def createA():
    # sorting as in cardillo
    # q = (rx0, ... rxN, ry0, ... ryN, rz0, ... rzN, Pr0, ... PrN, Px0, ... PxN, Py0, ... PyN, Pz0, ... PzN)
    # qe = (rx0, ... rxp, ry0, ... ryp, rz0, ... rzp, Pr0, ... Prp, Px0, ... Pxp, Py0, ... Pyp, Pz0, ... Pzp)
    nodal = np.arange(0, nq_node) * nnodes
    qDOF_node = np.array([nodal + node for node in range(nnodes)])
    nodalDOF_element = np.array(
        [np.arange(0, nq_node) * nnodes_element + node for node in range(p + 1)]
    )
    nodal_element = np.concatenate(
        [np.arange(p + 1) + nnodes * i for i in range(nq_node)]
    )
    qDOF_element = np.array([nodal_element + p * el for el in range(nelements)])

    return dict(
        nodalDOF_element=nodalDOF_element,
        qDOF_node=qDOF_node,
        qDOF_element=qDOF_element,
    )


def createB(sliced=False):
    # q = (r0, P0, ... rN, PN)
    # qe = (r0, P0, ... rp, Pp)
    if sliced:
        qDOF_node = [
            slice(node * nq_node, (node + 1) * nq_node) for node in range(nnodes)
        ]
        nodalDOF_element = [
            slice(node * nq_node, (node + 1) * nq_node)
            for node in range(nnodes_element)
        ]
        qDOF_element = [
            slice(el * p * nq_node, el * p * nq_node + nnodes_element * nq_node)
            for el in range(nelements)
        ]
    else:
        qDOF_node = np.array(
            [np.arange(0, nq_node) + node * nq_node for node in range(nnodes)]
        )
        nodalDOF_element = np.array(
            [np.arange(0, nq_node) + node * nq_node for node in range(p + 1)]
        )
        qDOF_element = np.array(
            [
                np.arange(0, (p + 1) * nq_node) + p * nq_node * el
                for el in range(nelements)
            ]
        )

    return dict(
        nodalDOF_element=nodalDOF_element,
        qDOF_node=qDOF_node,
        qDOF_element=qDOF_element,
    )


def test_correctness(q, sort):
    qDOF_element = sort["qDOF_element"]
    qDOF_node = sort["qDOF_node"]
    nodalDOF_element = sort["nodalDOF_element"]

    e = 0.0

    for element in range(nelements):
        qe = q[qDOF_element[element]]

        for node in range(p + 1):
            qNode = qe[nodalDOF_element[node]]

            node_global = element * p + node

            e += np.linalg.norm(qNode - q[qDOF_node[node_global]])

    return e


# test random q
sorting_cardillo = createA()
sorting_continuous = createB(sliced=False)
sorting_sliced = createB(sliced=True)
q = np.random.rand(nq)

nqaudarture_int = p
N_int = np.random.rand(nqaudarture_int * nnodes_element).reshape(
    [nqaudarture_int, nnodes_element]
)
wpi_int = np.random.rand(nqaudarture_int)


print(test_correctness(q, sorting_cardillo))
print(test_correctness(q, sorting_continuous))
print(test_correctness(q, sorting_sliced))


def perf_test(sort):
    qDOF_element = sort["qDOF_element"]
    qDOF_node = sort["qDOF_node"]
    nodalDOF_element = sort["nodalDOF_element"]

    def test_():
        f = np.zeros(nq)
        for el in range(nelements):
            f_el = np.zeros(nq_element)
            elDOF = qDOF_element[el]
            qe = q[elDOF]

            for qp in range(nqaudarture_int):
                q_inter = np.zeros(nq_node)
                for node in range(nnodes_element):
                    nodalDOF = nodalDOF_element[node]
                    q_node = qe[nodalDOF]
                    q_inter += q_node * N_int[qp, node]

                # do something with interpolated quantity
                fi = q_inter.copy()
                fi[0] += fi[2] * 2.0

                for node in range(nnodes_element):
                    f_el[nodalDOF] += fi * wpi_int[qp]

            f[elDOF] += f_el

        for node in range(nnodes):
            nodalDOF = qDOF_node[node]
            q_node = q[nodalDOF]
            f[nodalDOF] -= q_node * 5.0
        return f

    return test_


def perf_test_vector(sort):
    qDOF_element = sort["qDOF_element"]
    qDOF_node = sort["qDOF_node"]
    nodalDOF_element = sort["nodalDOF_element"]

    def test_():
        f = np.zeros(nq)
        for el in range(nelements):
            elDOF = qDOF_element[el]
            qe_nodes = q[elDOF].reshape(nnodes_element, nq_node)

            # Interpolation an allen Quadraturpunkten
            q_inter_all = N_int @ qe_nodes  # (nqp, nq_node)

            # Elementphysik
            fi_all = q_inter_all.copy()
            fi_all[:, 0] += 2.0 * fi_all[:, 2]

            # Rückprojektion
            f_el_nodes = N_int.T @ (fi_all * wpi_int[:, None])

            f[elDOF] += f_el_nodes.reshape(nq_element)

        for node in range(nnodes):
            nodalDOF = qDOF_node[node]
            q_node = q[nodalDOF]
            f[nodalDOF] -= q_node * 5.0
        return f

    return test_


# run once
perf_test(sorting_cardillo)
perf_test(sorting_continuous)
perf_test(sorting_sliced)
perf_test_vector(sorting_cardillo)
perf_test_vector(sorting_continuous)
perf_test_vector(sorting_sliced)

res0 = perf_test(sorting_continuous)()
res1 = perf_test(sorting_sliced)()
res2 = perf_test(sorting_sliced)()
print(np.linalg.norm(res1 - res0))
print(np.linalg.norm(res1 - res2))

ntest = 1_000
print(
    f"Cardillo            : {timeit.timeit(perf_test(sorting_cardillo), number=ntest)}"
)
print(
    f"Continous           : {timeit.timeit(perf_test(sorting_continuous), number=ntest)}"
)
print(f"Cont. slice         : {timeit.timeit(perf_test(sorting_sliced), number=ntest)}")
print(
    f"Vector - Cardillo   : {timeit.timeit(perf_test_vector(sorting_cardillo), number=ntest)}"
)
print(
    f"Vector - Continous  : {timeit.timeit(perf_test_vector(sorting_continuous), number=ntest)}"
)
print(
    f"Vector - Cont. slice: {timeit.timeit(perf_test_vector(sorting_sliced), number=ntest)}"
)
