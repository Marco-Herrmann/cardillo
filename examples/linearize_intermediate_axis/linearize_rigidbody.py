import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pprint import pprint
import scipy
import scipy.linalg

from cardillo import System
from cardillo.discrete import Box, PointMass, Frame, RigidBody
from cardillo.force_laws import KelvinVoigtElement as SpringDamper
from cardillo.interactions import TwoPointInteraction
from cardillo.math import Exp_SO3_quat, e3, ax2skew, cross3, ei
from cardillo.math.approx_fprime import approx_fprime
from cardillo.solver import ScipyIVP, Solution


def scipy_eig(*args, **kwargs):
    eig = scipy.linalg.eig(*args, **kwargs)
    if eig[-1].dtype == complex:
        return eig
    elif eig[-1].dtype == float:
        eig = list(eig)
        assert np.isclose(np.linalg.norm(np.imag(eig[0])), 0.0)
        eig[0] = eig[0].astype(float)
        return tuple(eig)


if __name__ == "__main__":
    ###################
    # system parameters
    ###################
    block_dim = np.array([5.0, 3.0, 2.0])
    omega = 4 * np.pi
    axis = 1

    # initial conditions
    r_OC = np.array([0, 0, 0], dtype=float)
    A_IB = np.eye(3, dtype=float)
    # A_IB = Exp_SO3_quat(np.array([1.0, 2.0, 3.0, 4.0]), normalize=True)
    # A_IB = Exp_SO3_quat(2 * np.random.rand(4) - 1, normalize=True)

    # initialize rigid body
    B_ei_B = ei(axis)
    q0_init = RigidBody.pose2q(r_OC, A_IB)
    u0_init = np.zeros(6, dtype=float)
    u0_init[3:] += omega * B_ei_B
    block = Box(RigidBody)(
        dimensions=block_dim,
        density=0.1,
        q0=q0_init,
        u0=u0_init,
        name="block",
    )

    #################
    # assemble system
    #################

    # initialize system
    system = System()
    system.add(block)
    system.assemble()

    # compute matrices
    # equations in form:    M @ u_dot = h, q_dot = B @ u
    # linearization:        M @ u_dot = h_q @ q_dot + h_u @ u, q_dot = B @ u
    # linear system: M @ u_dot + D @ u + K_bar @ q =0
    # with      D = -h_u, K_bar = d(M @ u0_dot)/dq - h_q
    P0_cos = system.q0[3:]
    P0_cos_0 = P0_cos[0]
    P0_cos_r = P0_cos[1:]
    P0_sin_0 = -P0_cos_r @ B_ei_B
    P0_sin_r = P0_cos_0 * B_ei_B + cross3(P0_cos_r, B_ei_B)
    P0_sin = np.concatenate([[P0_sin_0], P0_sin_r])

    r_OP0 = system.q0[:3]
    P0 = lambda t: (
        P0_cos * np.cos(1 / 2 * omega * t) + P0_sin * np.sin(1 / 2 * omega * t)
    )
    q0 = lambda t: np.concatenate([r_OP0, P0(t)])
    u0 = lambda t: system.u0
    u0_dot = lambda t: system.u_dot0

    if False:
        # numerical validation of analytical q0
        ts = np.linspace(0, 1, 500)
        # simple Euler forward integration with normalization
        qs = np.zeros([len(ts), system.nq], dtype=float)
        qs[0] = system.q0
        for i, ti in enumerate(ts[:-1]):
            q = qs[i]
            q_dot = system.q_dot(ti, q, system.u0)
            q_new = q + q_dot * ts[1]
            r_new = q_new[:3]
            p_new = q_new[3:]
            qs[i + 1] = np.concatenate([r_new, p_new / np.linalg.norm(p_new)])

        q0_analytical = np.array([q0(ti) for ti in ts])

        fig, ax = plt.subplots(4)
        ax[0].plot(ts, qs[:, 3])
        ax[1].plot(ts, qs[:, 4])
        ax[2].plot(ts, qs[:, 5])
        ax[3].plot(ts, qs[:, 6])

        ax[0].plot(ts, q0_analytical[:, 3], "--")
        ax[1].plot(ts, q0_analytical[:, 4], "--")
        ax[2].plot(ts, q0_analytical[:, 5], "--")
        ax[3].plot(ts, q0_analytical[:, 6], "--")

        for axi in ax:
            axi.grid()

        plt.show()

    # matrices from linearization
    M0 = lambda t: system.M(t, q0(t)).toarray()  # constant
    K0_bar = (
        lambda t: system.Mu_q(t, q0(t), u0_dot(t)).toarray()
        - system.h_q(t, q0(t), u0(t)).toarray()
    )  # zero
    D0_bar = lambda t: -system.h_u(t, q0(t), u0(t)).toarray()  # constnat

    # matrices for projection
    # dq = B @ dz
    # du = dz_dot + G @ dz
    # du_dot = dz_ddot + G @ dz_dot + G_dot @ dz
    B0 = lambda t: system.q_dot_u(t, q0(t)).toarray()

    def G0(t):  # constant
        mtx = np.zeros((system.nu, system.nu))
        mtx[3:, 3:] = ax2skew(u0(t)[3:])
        return mtx

    def G0_dot(t):  # zero
        mtx = np.zeros((system.nu, system.nu))
        mtx[3:, 3:] = ax2skew(u0_dot(t)[3:])
        return mtx

    # do the projection
    D0 = lambda t: D0_bar(t) + M0(t) @ G0(t)  # constant
    K0 = lambda t: K0_bar(t) @ B0(t) + D0_bar(t) @ G0(t) + M0(t) @ G0_dot(t)  # constant

    # as the matrices are constant, we can evaluate them at 0
    t0 = 0.0
    M00 = M0(t0)
    D00 = D0(t0)
    K00 = K0(t0)

    # compose matrices for EVP
    null_66 = np.zeros([6, 6], dtype=float)
    A_hat = np.block([[-K00, null_66], [null_66, M00]])
    B_hat = np.block([[D00, M00], [M00, null_66]])

    # determine eigenvalues
    las_g, Vs_g = scipy_eig(A_hat, B_hat)

    # only do rotational DOFs
    null_33 = np.zeros([3, 3], dtype=float)
    A_hat_rot = np.block([[-K00[3:, 3:], null_33], [null_33, M00[3:, 3:]]])
    B_hat_rot = np.block([[D00[3:, 3:], M00[3:, 3:]], [M00[3:, 3:], null_33]])

    las_rot, Vs_rot = scipy_eig(A_hat_rot, B_hat_rot)

    # print results
    ABC = np.diag(M00)[3:]
    print(f"Theta values: {ABC}, axis of rotation: {axis}")
    rot_theta = ABC[axis]
    others = np.delete(ABC, axis)
    if rot_theta >= np.max(others) or rot_theta <= np.min(others):
        stable = True
        print("Axis should be stable!")
    else:
        stable = False
        print("Axis should be unstable!")

    print("Computed values: ")
    
    eps = 1e-6
    for las in [las_g, las_rot]:
        print(
            f"lambdas: {[str('{s:.3f} + {r:.3f}i').format(s=np.real(lai), r=np.imag(lai)) for lai in las]}"
        )

        n_l0 = np.sum(np.real(las_rot) > eps)
        if stable:
            assert n_l0 == 0, "There should be no eigenvalue with positive real part."
        else:
            assert n_l0 == 1, "There should be one eigenvalue with positive real part."

    print("\n       Prima\n")
