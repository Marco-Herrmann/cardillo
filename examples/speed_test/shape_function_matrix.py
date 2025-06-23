import numpy as np
from scipy.sparse import diags_array
from timeit import timeit


from cardillo.rods.discretization import Mesh1D, LagrangeKnotVector
from cardillo.utility.coo_matrix import CooMatrix

p, nel = 5, 50

kv = LagrangeKnotVector(p, nel)
mesh = Mesh1D(kv, p + 1, 7, 1, "Lagrange", "Gauss", 6)

nq = mesh.nq
nq_element = mesh.nq_per_element

nquadrature = mesh.nquadrature
qp = mesh.qp
wp = mesh.wp


# build element matrix with shape functions
N_el = np.zeros((nel, nquadrature, 7, nq_element), dtype=float)
N_global = np.zeros((nel, nquadrature, 7, nq), dtype=float)
N_el_sparse_csr = np.zeros((nel, nquadrature), dtype=object)
N_el_sparse_csc = np.zeros((nel, nquadrature), dtype=object)
N_sparse_csr = np.zeros((nel, nquadrature), dtype=object)
N_sparse_csc = np.zeros((nel, nquadrature), dtype=object)
for el in range(nel):
    for i in range(nquadrature):
        N_eli_sparse = CooMatrix((7, nq_element))
        N_i_sparse = CooMatrix((7, nq))
        for node in range(p + 1):
            nodalDOF = mesh.nodalDOF_element[node]
            nodalDOF_global = mesh.elDOF[el][nodalDOF]

            dense_diag = np.eye(7, dtype=float) * mesh.N[el, i, node]
            N_el[el, i, :, nodalDOF] = dense_diag
            N_global[el, i, :, nodalDOF_global] = dense_diag

            sparse_diag = diags_array(np.ones(7, dtype=float) * mesh.N[el, i, node])
            N_eli_sparse[:, nodalDOF] = sparse_diag
            N_i_sparse[:, nodalDOF_global] = sparse_diag

        N_el_sparse_csr[el, i] = N_eli_sparse.tocsr()
        N_el_sparse_csc[el, i] = N_eli_sparse.tocsc()

        N_sparse_csr[el, i] = N_i_sparse.tocsr()
        N_sparse_csc[el, i] = N_i_sparse.tocsc()


def element_implementation(q):
    blub_result = np.zeros(7, dtype=float)
    for el in range(nel):
        qe = q[mesh.elDOF[el]]
        for i in range(nquadrature):
            blub = N_el[el, i] @ qe
            blub_result += blub * wp[el, i]

    return blub_result


def global_implementation(q):
    blub_result = np.zeros(7, dtype=float)
    for el in range(nel):
        for i in range(nquadrature):
            blub = N_global[el, i] @ q
            blub_result += blub * wp[el, i]

    return blub_result


def element_implementation_sparse_csc(q):
    blub_result = np.zeros(7, dtype=float)
    for el in range(nel):
        qe = q[mesh.elDOF[el]]
        for i in range(nquadrature):
            blub = N_el_sparse_csc[el, i] @ qe
            blub_result += blub * wp[el, i]

    return blub_result


def element_implementation_sparse_csr(q):
    blub_result = np.zeros(7, dtype=float)
    for el in range(nel):
        qe = q[mesh.elDOF[el]]
        for i in range(nquadrature):
            blub = N_el_sparse_csr[el, i] @ qe
            blub_result += blub * wp[el, i]

    return blub_result


def global_implementation_sparse_csc(q):
    blub_result = np.zeros(7, dtype=float)
    for el in range(nel):
        for i in range(nquadrature):
            blub = N_sparse_csc[el, i] @ q
            blub_result += blub * wp[el, i]

    return blub_result


def global_implementation_sparse_csr(q):
    blub_result = np.zeros(7, dtype=float)
    for el in range(nel):
        for i in range(nquadrature):
            blub = N_sparse_csr[el, i] @ q
            blub_result += blub * wp[el, i]

    return blub_result


def current_implementation(q):
    blub_result = np.zeros(7, dtype=float)
    for el in range(nel):
        qe = q[mesh.elDOF[el]]
        for i in range(nquadrature):
            blub = np.zeros(7, dtype=float)
            for node in range(p + 1):
                blub += mesh.N[el, i, node] * qe[mesh.nodalDOF_element[node]]

            blub_result += blub * wp[el, i]

    return blub_result


def tianxiang_shortening(q):
    blub_result = np.zeros(7, dtype=float)
    for el in range(nel):
        qe = q[mesh.elDOF[el]]
        for i in range(nquadrature):
            blub = mesh.N[el, i] @ qe[mesh.nodalDOF_element]

            blub_result += blub * wp[el, i]

    return blub_result


funcs = [
    current_implementation,
    element_implementation,
    element_implementation_sparse_csr,
    element_implementation_sparse_csc,
    global_implementation,
    global_implementation_sparse_csc,
    global_implementation_sparse_csr,
    tianxiang_shortening,
]
q = np.random.rand(nq)

nfcts = len(funcs)
func_names = [str(fun)[10:-23] for fun in funcs]
max_len = max([len(name) for name in func_names])
func_names_format = [f"{name:<{max_len+1}}" for name in func_names]

# check result
np.set_printoptions(linewidth=250)
for fun, name in zip(funcs, func_names_format):
    print(f"{name}: {fun(q)}")


# test time result
num = 500
for fun, name in zip(funcs, func_names_format):
    tmt = timeit("fun(q)", globals=globals(), number=500)
    print(f"{name}: {tmt}")
