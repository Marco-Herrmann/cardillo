import numpy as np
from scipy.sparse import csr_matrix, bsr_array, bsr_matrix
import timeit

nelement = 500
p = 2
nnodes = nelement * p + 1
nquadrature = p + 1
nquadrature = p
nq_node = 7
nu_node = 6


nq = nnodes * nq_node
nu = nnodes * nu_node

q = np.random.rand(nq)
u = np.random.rand(nu)


# interpolation matrices
N = np.zeros((nquadrature * nelement, nnodes))
N_xi = np.zeros((nquadrature * nelement, nnodes))

# # these should be really dense, but for testing we can just use random matrices
# N = np.random.rand(nquadrature * nelements * nnodes).reshape(nquadrature * nelements, nnodes)
# N_xi = np.random.rand(nquadrature * nelements * nnodes).reshape(nquadrature * nelements, nnodes)

N_sparse = csr_matrix((nquadrature * nelement, nnodes))
N_xi_sparse = csr_matrix((nquadrature * nelement, nnodes))

for el in range(nelement):
    nodes = np.arange(el * p, el * p + p + 1)
    for i in range(nquadrature):
        for node in nodes:
            idx = el * nquadrature + i, node

            val_N = np.random.rand()
            val_N_xi = np.random.rand()

            N[idx] = val_N
            N_xi[idx] = val_N_xi

            N_sparse[idx] = val_N
            N_xi_sparse[idx] = val_N_xi


def get_all():
    rP_node = q.reshape(nnodes, nq_node)

    rP_qp = N @ rP_node
    rP_qp_xi = N_xi @ rP_node


def get_all_sparse():
    rP_node = q.reshape(nnodes, nq_node)

    rP_qp = N_sparse @ rP_node
    rP_qp_xi = N_xi_sparse @ rP_node


def get_all_loop():
    rP_node = q.reshape(nnodes, nq_node)

    for el in range(nelement):
        for i in range(nquadrature):
            rP_qp = np.zeros(nq_node)
            rP_qp_xi = np.zeros(nq_node)
            for node in range(p + 1):
                rP_qp += N[i, node] * rP_node[node]
                rP_qp_xi += N_xi[i, node] * rP_node[node]


def get_all_p1():
    rP_node = q.reshape(nnodes, nq_node)

    rP_qp = (rP_node[1:] + rP_node[:-1]) / 2
    rP_qp_xi = (rP_node[1:] - rP_node[:-1]) / 2


qDOF = np.arange(nq_node)


def get_all_fake():
    for el in range(nelement):
        for qp in range(nquadrature):
            rP_qp = np.zeros(nq_node)
            rP_qp_xi = np.zeros(nq_node)
            for node in range(p + 1):
                qNode = q[qDOF]
                rP_qp += qNode
                rP_qp_xi += qNode


print(f"{np.shares_memory(q, q.reshape(nnodes, nq_node))=}")


if __name__ == "__main__":
    get_all()
    get_all_sparse()
    get_all_loop()
    get_all_p1()
    get_all_fake()

    ntest = 1_000
    print(
        f"Time for dense interpolation : {timeit.timeit(get_all, number=ntest)} seconds"
    )
    print(
        f"Time for sparse interpolation: {timeit.timeit(get_all_sparse, number=ntest)} seconds"
    )
    print(
        f"Time for loop interpolation  : {timeit.timeit(get_all_loop, number=ntest)} seconds"
    )
    print(
        f"Time for p1 interpolation    : {timeit.timeit(get_all_p1, number=ntest)} seconds"
    )
    print(
        f"Time for fake interpolation  : {timeit.timeit(get_all_fake, number=ntest)} seconds"
    )
