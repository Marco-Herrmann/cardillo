import numpy as np
import timeit
from scipy.sparse import csr_matrix, block_diag, lil_matrix, bsr_matrix

from cardillo.math.rotations import (
    Exp_SO3_quat,
    T_SO3_quat,
    Exp_SO3_quat_P,
    T_SO3_quat_P,
)
from cardillo.math import ax2skew, ax2skew_squared


def ax2skew_multi(a: np.ndarray):
    was_1d = a.ndim == 1
    S = np.array([ax2skew(ai) for ai in np.atleast_2d(a)])
    return S[0] if was_1d else S


def ax2skew_new(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    was_1d = a.ndim == 1
    a = np.atleast_2d(a)
    assert a.shape[1] == 3

    # fmt: off
    S = np.zeros((a.shape[0], 3, 3), dtype=a.dtype)
    S[:, 0, 1] = -a[:, 2]
    S[:, 0, 2] =  a[:, 1]
    S[:, 1, 0] =  a[:, 2]
    S[:, 1, 2] = -a[:, 0]
    S[:, 2, 0] = -a[:, 1]
    S[:, 2, 1] =  a[:, 0]
    # fmt: on

    return S[0] if was_1d else S


def ax2skew_squared_multi(a: np.ndarray):
    was_1d = a.ndim == 1
    S2 = np.array([ax2skew_squared(ai) for ai in np.atleast_2d(a)])
    return S2[0] if was_1d else S2


def ax2skew_squared_new(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    was_1d = a.ndim == 1
    a = np.atleast_2d(a)
    assert a.shape[1] == 3

    a1, a2, a3 = a[:, 0], a[:, 1], a[:, 2]
    a1a1 = a1 * a1
    a2a2 = a2 * a2
    a3a3 = a3 * a3

    S = np.empty((a.shape[0], 3, 3), dtype=a.dtype)
    S[:, 0, 0] = -(a2a2 + a3a3)
    S[:, 1, 1] = -(a1a1 + a3a3)
    S[:, 2, 2] = -(a1a1 + a2a2)

    S[:, 0, 1] = S[:, 1, 0] = a1 * a2
    S[:, 0, 2] = S[:, 2, 0] = a1 * a3
    S[:, 1, 2] = S[:, 2, 1] = a2 * a3

    return S[0] if was_1d else S


def Exp_SO3_quat_multi(P_IB: np.ndarray, normalize: bool = True):
    was_1d = P_IB.ndim == 1
    A_IBs = np.array([Exp_SO3_quat(Pi, normalize) for Pi in np.atleast_2d(P_IB)])
    return A_IBs[0] if was_1d else A_IBs


eye3 = np.eye(3, dtype=float)


def Exp_SO3_quat_new(P: np.ndarray, normalize: bool = True) -> np.ndarray:
    P = np.asarray(P)
    was_1d = P.ndim == 1
    P = np.atleast_2d(P)
    assert P.shape[1] == 4

    p0, p = P[:, 0], P[:, 1:]
    matrix = 2.0 * (p0[:, None, None] * ax2skew_new(p) + ax2skew_squared_new(p))
    if normalize:
        matrix /= np.sum(P * P, axis=1)[:, None, None]

    result = eye3[None, :, :] + matrix
    return result[0] if was_1d else result


def T_SO3_quat_multi(P_IB: np.ndarray, normalize: bool = True):
    was_1d = P_IB.ndim == 1
    Ts = np.array([T_SO3_quat(Pi, normalize) for Pi in np.atleast_2d(P_IB)])
    return Ts[0] if was_1d else Ts


def T_SO3_quat_new(P_IB: np.ndarray, normalize: bool = True) -> np.ndarray:
    P_IB = np.asarray(P_IB)
    was_1d = P_IB.ndim == 1
    P_IB = np.atleast_2d(P_IB)
    assert P_IB.shape[1] == 4

    p0, p = P_IB[:, 0], P_IB[:, 1:]
    result = np.empty((P_IB.shape[0], 3, 4), dtype=np.result_type(P_IB, 1.0))
    result[:, :, 0] = -2.0 * p
    result[:, :, 1:] = 2.0 * (p0[:, None, None] * eye3[None, :, :] - ax2skew_new(p))
    if normalize:
        result /= np.sum(P_IB * P_IB, axis=1)[:, None, None]
    return result[0] if was_1d else result


#########
# jaxen #
#########
import jax
from jax import numpy as jnp
from jax import vmap, jit

jax.config.update("jax_enable_x64", True)


@jit
def ax2skew_jax(a: jnp.ndarray) -> jnp.ndarray:
    """Computes the skew symmetric matrix from a 3D vector."""
    # fmt: off
    return jnp.array([[0,    -a[2], a[1] ],
                      [a[2],  0,    -a[0]],
                      [-a[1], a[0], 0    ]], dtype=jnp.float64)
    # fmt: on


@jit
def ax2skew_squared_jax(a: jnp.ndarray) -> jnp.ndarray:
    """Computes the product of a skew-symmetric matrix with itself from a given axial vector."""
    a1, a2, a3 = a
    # fmt: off
    return jnp.array([
        [-a2**2 - a3**2,              a1 * a2,              a1 * a3],
        [             a2 * a1, -a1**2 - a3**2,              a2 * a3],
        [             a3 * a1,              a3 * a2, -a1**2 - a2**2],
    ], dtype=jnp.float64)
    # fmt: on


@jit
def Exp_SO3_quat_jax(P, normalize: bool = True):
    p0, p = P[0], P[1:]

    return jnp.where(
        normalize,
        eye3 + (2.0 / (P @ P)) * (p0 * ax2skew_jax(p) + ax2skew_squared_jax(p)),
        eye3 + 2.0 * (p0 * ax2skew_jax(p) + ax2skew_squared_jax(p)),
    )


ax2skew_batch = jit(vmap(ax2skew_jax))
Exp_SO3_quat_batch = jit(vmap(Exp_SO3_quat_jax, in_axes=(0, None)))
# Exp_SO3_quat_batch = jit(vmap(lambda P: Exp_SO3_quat_jax(P, True)))

n_points = 500
P_IBs = np.random.rand(4 * n_points).reshape(n_points, 4)

A_IB0 = Exp_SO3_quat_multi(P_IBs)
A_IB1 = Exp_SO3_quat_new(P_IBs)
A_IB2 = Exp_SO3_quat_batch(P_IBs, True)

# timeit
ntimeit = 1_000
t1 = timeit.timeit(lambda: Exp_SO3_quat_multi(P_IBs), number=ntimeit)
t2 = timeit.timeit(lambda: Exp_SO3_quat_new(P_IBs), number=ntimeit)
t3 = timeit.timeit(lambda: np.asarray(Exp_SO3_quat_batch(P_IBs, True)), number=ntimeit)
print(f"Time for loop: {t1}")
print(f"Time Vector  : {t2}")
print(f"Time jax     : {t3}")

print(f"ratios: loop/jax: {t1 / t3}, vec/jax: {t2 / t3}, loop/vec: {t1 / t2}")


exit()

###################
# test balken stuff
###################

L = 1.0
nelements = 100  # equal runtime around 4-5 elements!
nnodes = nelements + 1
nq_node = 7
nu_node = 6

C_gamma = np.diag([1.0, 2.0, 3.0])
C_kappa = np.diag([1.0, 2.0, 3.0])
C = np.diag([*1 / np.diag(C_gamma), *1 / np.diag(C_kappa)])
gammas0 = np.broadcast_to(
    [1, 0, 0], (nelements, 3)
)  # Note: get them from Q (q0), but thats the shape they need
kappas0 = np.broadcast_to(
    [0, 0, 0], (nelements, 3)
)  # Note: get them from Q (q0), but thats the shape they need
Li = np.array(
    nelements * [L / nelements]
)  # Note: get this also from Q (q0), but shape-wise it should work like this
Cs_gamma = np.broadcast_to(
    C_gamma, (nelements, 3, 3)
)  # Note: this could also be computed per quadrature point or per element dependent on xi
Cs_kappa = np.broadcast_to(
    C_kappa, (nelements, 3, 3)
)  # Note: this could also be computed per quadrature point or per element dependent on xi
Cs = np.broadcast_to(
    C, (nelements, 6, 6)
)  # Note: this could also be computed per quadrature point or per element dependent on xi
Li_inv = 1 / Li

nq = nnodes * nq_node
nu = nnodes * nu_node
nla_c = nelements * 6
q = np.random.rand(nq)


def f_int(q_, compliance=True):
    # TODO: somehwere division/multiplication by L
    rPs = q_.reshape([nnodes, nq_node])

    rPs_mid = (rPs[:-1] + rPs[1:]) / 2
    drPs_mid = (rPs[1:] - rPs[:-1]) / 2  # is the 2 wrong?

    A_IBs = Exp_SO3_quat_new(rPs_mid[:, 3:])
    Ts = T_SO3_quat_new(rPs_mid[:, 3:])

    gammas_bar = np.einsum("ijk,ij->ik", A_IBs, drPs_mid[:, :3])
    kappas_bar = np.einsum("ijk,ik->ij", Ts, drPs_mid[:, 3:])

    epsilon_ga = gammas_bar * Li_inv[:, None] - gammas0
    epsilon_ka = kappas_bar * Li_inv[:, None] - kappas0

    # TODO: can we use Cs as (nelement, 6, 6) instead? or use prismatic==True and simpleQUadratic_constitutive_law == True
    # probably not that relevant, as for compliance form, we have to fill the whole matrix anyways, or we have to implement
    # def Wla_c(t, q, la_c) ... directly to simplify again
    # then it would be
    # B_ns = (Cs_gamma @ epsilon_ga[:, :, None]).squeeze()
    # B_ms = (Cs_kappa @ epsilon_ka[:, :, None]).squeeze()
    # TODO: would be cool to have time-dependent stiffness (atleast for the complex case with xi dependency --> for my rope-simulation)
    B_ns = epsilon_ga @ C_gamma.T
    B_ms = epsilon_ka @ C_kappa.T

    gamma_bar_cross_n = np.cross(gammas_bar, B_ns)
    kappa_bar_cross_m = np.cross(kappas_bar, B_ms)

    ns = np.einsum("ijk,ik->ij", A_IBs, B_ns)

    # check how this reshape works
    nm = np.concatenate([ns, B_ms], axis=1).reshape(-1)
    tilde_thing = np.concatenate(
        [np.zeros_like(gamma_bar_cross_n), gamma_bar_cross_n + kappa_bar_cross_m],
        axis=1,
    ).reshape(-1)

    # check factor, sign, weights, etc.
    f_int = np.zeros(nu)
    f_int[nu_node:] += tilde_thing + nm
    f_int[:-nu_node] += tilde_thing - nm

    ####################
    # compliance stuff #
    ####################
    if compliance:
        gammas_bar_tilde = ax2skew_new(gammas_bar)
        kappas_bar_tilde = ax2skew_new(kappas_bar)
        # check order of reshape
        tilde_thing_W = np.empty((nelements, 6, 6))
        tilde_thing_W[:, :3, :] = 0.0
        tilde_thing_W[:, 3:, :3] = gammas_bar_tilde
        tilde_thing_W[:, 3:, 3:] = kappas_bar_tilde

        # TODO: check signs
        tilde_thing_diag = bsr_matrix(
            (tilde_thing_W, np.arange(nelements), np.arange(nelements + 1)),
            shape=(6 * nelements, 6 * nelements),
        )

        rest_thing_W = np.empty((nelements, 6, 6))
        rest_thing_W[:, :3, :3] = A_IBs
        rest_thing_W[:, :3, 3:] = 0.0
        rest_thing_W[:, 3:, :3] = 0.0
        rest_thing_W[:, 3:, 3:] = eye3[None, :, :]

        rest_thing_diag = bsr_matrix(
            (rest_thing_W, np.arange(nelements), np.arange(nelements + 1)),
            shape=(6 * nelements, 6 * nelements),
        )

        # # oh wow, lil matrix seams to be slow as fuck
        # # check factor, sign, weights, etc.
        # W_c = lil_matrix((nu, nla_c))
        # W_c[nu_node:, :] += tilde_thing_diag + rest_thing_diag
        # W_c[:-nu_node, :] += tilde_thing_diag - rest_thing_diag

        # new way using directly sctructure of csr
        ttd_coo = tilde_thing_diag.tocoo()
        rtd_coo = rest_thing_diag.tocoo()

        assert np.linalg.norm(ttd_coo.row - rtd_coo.row) == 0
        assert np.linalg.norm(ttd_coo.col - rtd_coo.col) == 0

        rows = np.concatenate([ttd_coo.row, ttd_coo.row + nu_node])
        cols = np.concatenate([ttd_coo.col, ttd_coo.col])
        data = np.concatenate(
            [ttd_coo.data - rtd_coo.data, ttd_coo.data + rtd_coo.data]
        )

        W_c_new = csr_matrix((data, (rows, cols)), shape=(nu, nla_c))
        K_c_inv = block_diag(Cs)  # preassembled
        c = np.concatenate([epsilon_ga, epsilon_ka], axis=1).reshape(
            -1
        )  # todo: check for speedup by not using concatenation and reshape

    return f_int


def f_int_old(q_):
    f_ = np.zeros(nu)
    for el in range(nelements):
        # for qp in range(1):
        r_OP = q_[:3]
        P_IB = q_[:4]
        r_OP_xi = q_[3:6]
        P_IB_xi = q_[3:7]
        A_IB = Exp_SO3_quat(P_IB)
        T = T_SO3_quat(P_IB)

        gamma = A_IB.T @ r_OP_xi
        kappa = T @ P_IB_xi

        f_[:3] += C_gamma @ gamma
        f_[3:6] += C_kappa @ kappa

    return f_


f_int(q)
f_int_old(q)


P_IBs = np.random.rand(nelements * 4).reshape([nelements, 4])
aas = np.random.rand(nelements * 3).reshape([nelements, 3])

np.set_printoptions(linewidth=300, precision=4)

# test ax2skew
# print(ax2skew(aas[0]))
# print(ax2skew_multi(aas[0]))
# print(ax2skew_multi(aas)[0])
# print(ax2skew_new(aas[0]))
# print(ax2skew_new(aas)[0])
print(np.linalg.norm(ax2skew_new(aas) - ax2skew_multi(aas)))

# test ax2skew_squared
# print(ax2skew_squared(aas[0]))
# print(ax2skew_squared_multi(aas[0]))
# print(ax2skew_squared_multi(aas)[0])
# print(ax2skew_squared_new(aas[0]))
# print(ax2skew_squared_new(aas)[0])
print(np.linalg.norm(ax2skew_squared_new(aas) - ax2skew_squared_multi(aas)))

# test Exp_SO3_quat
# print(Exp_SO3_quat(P_IBs[0]))
# print(Exp_SO3_quat_multi(P_IBs[0]))
# print(Exp_SO3_quat_multi(P_IBs)[0])
# print(Exp_SO3_quat_new(P_IBs[0]))
# print(Exp_SO3_quat_new(P_IBs)[0])
print(np.linalg.norm(Exp_SO3_quat_new(P_IBs) - Exp_SO3_quat_multi(P_IBs)))

# test T_SO3_quat
# print(T_SO3_quat(P_IBs[0]))
# print(T_SO3_quat_multi(P_IBs[0]))
# print(T_SO3_quat_multi(P_IBs)[0])
# print(T_SO3_quat_new(P_IBs[0]))
# print(T_SO3_quat_new(P_IBs)[0])
print(np.linalg.norm(T_SO3_quat_new(P_IBs) - T_SO3_quat_multi(P_IBs)))


# compare performance
ntest = 1_000
print(f"List-compr.: {timeit.timeit(lambda: T_SO3_quat_multi(P_IBs), number=ntest)}")
print(f"Vectorized : {timeit.timeit(lambda: T_SO3_quat_new(P_IBs), number=ntest)}")

ntest2 = 50
print(
    f"f + compliance: {timeit.timeit(lambda: f_int(q, compliance=True), number=ntest)}"
)
print(f"f_new: {timeit.timeit(lambda: f_int(q, compliance=False), number=ntest)}")
print(f"f_old: {timeit.timeit(lambda: f_int_old(q), number=ntest)}")
