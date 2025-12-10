import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial

from cardillo import System
from cardillo.solver import BackwardEuler, Newton, SolverOptions


from cardillo.math.rotations import (
    A_IB_basic,
    Exp_SO3_quat,
    Log_SO3_quat,
    T_SO3_quat,
    T_SO3_quat_P,
    T_SO3_inv_quat,
)
from cardillo.rods.discretization import gauss
from cardillo.math.approx_fprime import approx_fprime


def quat2angle(P):
    return np.atan2(2 * P[0] * P[1], P[0] ** 2 - P[1] ** 2)


def quat2angle_P(P):
    fct = 2 / (P[0] ** 2 + P[1] ** 2)
    return fct * np.array([-P[1], P[0]], dtype=float)


def quat2angle_PP(P):
    fct = 2 / (P[0] ** 2 + P[1] ** 2) ** 2
    diag = 2 * P[0] * P[1]
    off_diag = P[1] ** 2 - P[0] ** 2
    return fct * np.array([[diag, off_diag], [off_diag, -diag]], dtype=float)


def B_quat(P):
    return 1 / 2 * np.array([[-P[1], P[0]]], dtype=float)


class TorsionSpring:
    def __init__(
        self,
        *,
        GI=1.0,
        Irho0=1.0,
        L=1.0,
        normalize=True,
        nquadrature=1,
        polynomial_degree=1,
        q0=None,
        Q=None,
    ):
        self.GI = GI
        self.Irho0 = Irho0
        self.L = L

        nnodes = polynomial_degree + 1
        self.nodes = np.linspace(0, 1, nnodes)
        self.nq = 2 * nnodes
        self.nu = nnodes

        self.nla_S = nnodes

        self.q0 = np.array([1.0, 0.0] * nnodes, dtype=float) if q0 is None else q0
        self.u0 = np.array([0.0] * nnodes, dtype=float)

        self.nla_c = polynomial_degree
        self.la_c0 = np.array([0.0] * polynomial_degree)

        self.nquadrature = nquadrature
        self.qp, self.qw = gauss(self.nquadrature, [0, 1])

        # normalize in B
        self.normalize = normalize

        # precompute shape functions
        self.shape_functions = [self._sf(i) for i in range(polynomial_degree + 1)]

        # save
        self.polynomial_degree = polynomial_degree
        self.nnodes = nnodes
        self.Q = self.q0.copy() if Q is None else Q

        # clamping on the left side
        self.nla_g = 1

        # force on the right side
        self.moment_end = 0.0

    @staticmethod
    def twisted_configuration(*alphas):
        Ps = [Log_SO3_quat(A_IB_basic(alpha).x)[:2] for alpha in alphas]
        Ps = [Ps[0], *[np.sign(Ps[i] @ Ps[i - 1]) * Ps[i] for i in range(1, len(Ps))]]
        return np.concatenate(Ps)

    @staticmethod
    def linear_twist(alpha0, alpha1, polynomial_degree):
        diff = alpha1 - alpha0
        alphas = [
            alpha0 + diff * i / polynomial_degree for i in range(polynomial_degree + 1)
        ]
        return TorsionSpring.twisted_configuration(*alphas)

    # handling shape functions
    def _sf(self, p):
        nnodes = p + 1
        do_wi = [0, 1]
        kv = np.linspace(0, 1, nnodes)
        Ni = np.empty(nnodes, dtype=object)
        for i in range(nnodes):
            Ni[i] = Polynomial([1.0], domain=do_wi, window=do_wi)
            for j in range(nnodes):
                if i != j:
                    diff = kv[i] - kv[j]
                    Ni[i] *= Polynomial(
                        [-kv[j] / diff, 1.0 / diff], domain=do_wi, window=do_wi
                    )

        return Ni

    def sf(self, xi, p):
        return np.array([N(xi) for N in self.shape_functions[p]])

    def sf_xi(self, xi, p):
        return np.array([N.deriv(1)(xi) for N in self.shape_functions[p]])

    def _sf_reinterpolation(self):
        nnodes = self.nquadrature
        do_wi = [0, 1]
        kv = self.qp
        Ni = np.empty(nnodes, dtype=object)
        for i in range(nnodes):
            Ni[i] = Polynomial([1.0], domain=do_wi, window=do_wi)
            for j in range(nnodes):
                if i != j:
                    diff = kv[i] - kv[j]
                    Ni[i] *= Polynomial(
                        [-kv[j] / diff, 1.0 / diff], domain=do_wi, window=do_wi
                    )

        return Ni

    def sf_reinterpolation(self, xi):
        return np.array([N(xi) for N in self._sf_reinterpolation()])

    def Nq(self, xi):
        Nq = np.zeros((2, self.nq), dtype=float)
        N = self.sf(xi, self.polynomial_degree)
        for i, Ni in enumerate(N):
            Nq[:2, 2 * i : 2 * (i + 1)] = np.eye(2, dtype=float) * Ni
        return Nq

    def Nq_xi(self, xi):
        Nq_xi = np.zeros((2, self.nq), dtype=float)
        N_xi = self.sf_xi(xi, self.polynomial_degree)
        for i, Ni_xi in enumerate(N_xi):
            Nq_xi[:2, 2 * i : 2 * (i + 1)] = np.eye(2, dtype=float) * Ni_xi
        return Nq_xi

    def Nu(self, xi):
        Nu = np.zeros((1, self.nu), dtype=float)
        N = self.sf(xi, self.polynomial_degree)
        for i, Ni in enumerate(N):
            Nu[0, i] = Ni
        return Nu

    def Nu_xi(self, xi):
        Nu_xi = np.zeros((1, self.nu), dtype=float)
        N_xi = self.sf_xi(xi, self.polynomial_degree)
        for i, Ni_xi in enumerate(N_xi):
            Nu_xi[0, i] = Ni_xi
        return Nu_xi

    def Nla_c(self, xi):
        Nla_c = np.zeros((1, self.nla_c), dtype=float)
        N = self.sf(xi, self.polynomial_degree - 1)
        for i, Ni in enumerate(N):
            Nla_c[0, i] = Ni
        return Nla_c

    ######################
    # cardillo functions #
    ######################
    # kinematic differential equation
    def q_dot_u(self, t, q):
        q_dot_u = np.zeros((self.nq, self.nu), dtype=float)
        for i in range(self.nnodes):
            qDOF = np.array([2 * i, 2 * i + 1])
            q_dot_u[qDOF, i] = B_quat(self.Nq(self.nodes[i]) @ q)

        return q_dot_u

    # mass matrix
    def M(self, t, q):
        M = np.zeros((self.nu, self.nu), dtype=float)
        for i in range(self.nquadrature):
            qpi = self.qp[i]
            qwi = self.qw[i]

            Nu = self.Nu(qpi)
            M += Nu.T @ Nu * (self.Irho0 * qwi * self.L)

        return M

    # compliance
    def la_c(self, t, q):
        print("la_c is called")
        return np.linalg.solve(self.c_la_c(), self.l_c_master(q)[0])

    def l_c_master(self, q):
        l_c = np.zeros(self.nla_c, dtype=float)
        l_c_q = np.zeros((self.nla_c, self.nq), dtype=float)

        for i in range(self.nquadrature):
            qpi = self.qp[i]
            qwi = self.qw[i]

            kappa_bar0_x = self.kappa(qpi, self.Q)
            kappa_bar_x = self.kappa(qpi, q)
            kappa_bar_x_q = self.kappa_q(qpi, q)

            epsilon_T = np.atleast_1d(kappa_bar_x - kappa_bar0_x)
            epsilon_T_q = np.atleast_2d(kappa_bar_x_q)

            Nla_c = self.Nla_c(qpi)
            l_c += Nla_c.T @ epsilon_T * (self.L * qwi)
            l_c_q += Nla_c.T @ epsilon_T_q * (self.L * qwi)

        return l_c, l_c_q

    def c(self, t, q, u, la_c):
        return self.c_la_c() @ la_c - self.l_c_master(q)[0]

    def c_q(self, t, q, u=None, la_c=None):
        return -self.l_c_master(q)[1]

    def c_la_c(self):
        c_la_c = np.zeros((self.nla_c, self.nla_c), dtype=float)
        for i in range(self.nquadrature):
            qpi = self.qp[i]
            qwi = self.qw[i]

            Nla_c = self.Nla_c(qpi)
            c_la_c += Nla_c.T @ Nla_c * (qwi * self.L / self.GI)

        return c_la_c

    def W_c(self, t, q):
        W_c = np.zeros((self.nu, self.nla_c), dtype=float)
        for i in range(self.nquadrature):
            qpi = self.qp[i]
            qwi = self.qw[i]

            # no need to multiply/divide by the length
            Nu_xi = self.Nu_xi(qpi)
            Nla_c = self.Nla_c(qpi)
            W_c -= Nu_xi.T @ Nla_c * qwi

        return W_c

    def Wla_c_q(self, t, q, la_c):
        return None

    # quaternion constraint
    def g_S(self, t, q):
        g_S = np.zeros(self.nnodes)
        for i in range(self.nnodes):
            DOF = range(2 * i, 2 * (i + 1))
            g_S[i] = q[DOF] @ q[DOF] - 1.0

        return g_S

    def g_S_q(self, t, q):
        g_S_q = np.zeros((self.nnodes, self.nq), dtype=float)
        for i in range(self.nnodes):
            DOF = range(2 * i, 2 * (i + 1))
            g_S_q[i, DOF] = 2 * q[DOF]

        return g_S_q

    # clamping
    def g(self, t, q):
        return np.array([self.angle(0.0, q) - self.angle(0.0, self.Q)])

    def g_q(self, t, q):
        return approx_fprime(q, lambda q_: self.g(t, q_))

    def W_g(self, t, q):
        W_g = np.zeros([self.nu, self.nla_g], dtype=float)
        W_g[0, 0] = 1.0
        return W_g

    def Wla_g_q(self, t, q, la_g):
        return None

    # external load at xi=1
    def h(self, t, q, u):
        h = np.zeros(self.nu, dtype=float)
        h[-1] = self.moment_end * t
        return h

    # step callback
    def step_callback(self, t, q, u):
        for i in range(self.nnodes):
            DOF = range(2 * i, 2 * (i + 1))
            q[DOF] /= np.sqrt(q[DOF] @ q[DOF])

        return q, u

    ##################
    # eval functions #
    ##################
    def angle(self, xi, q):
        Pi = self.Nq(xi) @ q
        return quat2angle(Pi)

    def kappa(self, xi, q):
        P = self.Nq(xi) @ q
        P_xi = self.Nq_xi(xi) @ q
        return quat2angle_P(P) @ P_xi / self.L

    def kappa_q(self, xi, q):
        Nq = self.Nq(xi)
        Nq_xi = self.Nq_xi(xi)
        P = Nq @ q
        P_xi = Nq_xi @ q
        return (quat2angle_P(P) @ Nq_xi + P_xi @ quat2angle_PP(P) @ Nq) / self.L

    # Lagrange function
    def angle_Lagrange(self, xi, q):
        angles = np.array([self.angle(xii, q) for xii in self.nodes])
        return self.sf(xi, self.polynomial_degree) @ angles

    def kappa_Lagrange(self, xi, q):
        angles = np.array([self.angle(xii, q) for xii in self.nodes])
        return self.sf_xi(xi, self.polynomial_degree) @ angles  # * self.L

    # reinterpolation with quadrature points
    def angle_reinterpolate(self, xi, q):
        angles = np.array([self.angle(qpi, q) for qpi in self.qp])
        return self.sf_reinterpolation(xi) @ angles

    def kappa_reinterpolate(self, xi, q):
        kappas = np.array([self.kappa(qpi, q) for qpi in self.qp])
        return self.sf_reinterpolation(xi) @ kappas


def cardillo_linearize(start_angle, end_angle, moment_angle, G, rho, L):
    polynomial_degree = 6
    nquadrature = polynomial_degree  # reduced integration
    nquadrature = 2 * polynomial_degree

    q0 = TorsionSpring.linear_twist(start_angle, end_angle, polynomial_degree)
    spring = TorsionSpring(
        GI=G,
        Irho0=rho,
        L=L,
        nquadrature=nquadrature,
        polynomial_degree=polynomial_degree,
        q0=q0,
        Q=q0,
    )

    spring.moment_end = moment_angle * spring.GI

    # create system
    system = System()
    system.add(spring)

    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    n_steps = 1
    sol = Newton(system, n_steps, verbose=False).solve()
    q_linearize = sol.q[n_steps]
    omegas, modes_dq, sol_modes = system.new_eigenmodes(sol, n_steps)

    return omegas[0]

    print(f"solution: {q_linearize}")
    print(omegas)
    print(modes_dq)

    give_me_results(spring, q_linearize)


def give_me_results(spring: TorsionSpring, q: np.ndarray):
    # make unit quaternions
    np.set_printoptions(precision=5, linewidth=300)

    t = 0.0
    B = spring.q_dot_u(t, q)
    c_q = spring.c_q(t, q)
    W_c = spring.W_c(t, q)

    print(f"B: \n{B}")
    print(f"c_q: \n{c_q}")

    print(f"W_c: \n{W_c}")

    BT_cqT = B.T @ c_q.T
    print(f"B.T @ c_q.T: \n{BT_cqT}")

    diff = W_c - BT_cqT
    print(f"diff: \n{diff}")

    xis = np.linspace(0, 1, 201)
    P = np.array([spring.Nq(xi) @ q for xi in xis])

    alphas_quat = np.array([spring.angle(xi, q) for xi in xis])
    alphas_lagr = np.array([spring.angle_Lagrange(xi, q) for xi in xis])
    alphas_quat_ref = np.array([spring.angle(xi, spring.Q) for xi in xis])
    alphas_lagr_ref = np.array([spring.angle_Lagrange(xi, spring.Q) for xi in xis])

    kappas_quat = np.array([spring.kappa(xi, q) for xi in xis])
    kappas_lagr = np.array([spring.kappa_Lagrange(xi, q) for xi in xis])
    kappas_qp = np.array([spring.kappa(qpi, q) for qpi in spring.qp])
    kappas_rI = np.array([spring.kappa_reinterpolate(xi, q) for xi in xis])
    kappas_quat_ref = np.array([spring.kappa(xi, spring.Q) for xi in xis])
    kappas_lagr_ref = np.array([spring.kappa_Lagrange(xi, spring.Q) for xi in xis])
    kappas_qp_ref = np.array([spring.kappa(qpi, spring.Q) for qpi in spring.qp])
    kappas_rI_ref = np.array([spring.kappa_reinterpolate(xi, spring.Q) for xi in xis])

    fig, ax = plt.subplots(4, 2)
    # plot quaternion values
    ax[0, 0].plot(xis, P[:, 0], label="p0")
    ax[0, 0].plot(xis, P[:, 1], label="p1")

    # plot angle
    ax[1, 0].plot(xis, alphas_quat, label="alpha quaternion current")
    ax[1, 0].plot(xis, alphas_lagr, "--", label="alpha Lagrange current")

    ax[2, 0].plot(xis, alphas_quat_ref, label="alpha quaternion reference")
    ax[2, 0].plot(xis, alphas_lagr_ref, "--", label="alpha Lagrange reference")

    ax[3, 0].plot(xis, alphas_quat - alphas_quat_ref, label="alpha quaternion diff")
    ax[3, 0].plot(xis, alphas_lagr - alphas_lagr_ref, "--", label="alpha Lagrange diff")

    # plot torsion
    ax[1, 1].plot(xis, kappas_quat, label="kappa quaternion current")
    ax[1, 1].plot(xis, kappas_lagr, "--", label="kappa Lagrange current")
    ax[1, 1].plot(spring.qp, kappas_qp, "x", label="qpi")
    ax[1, 1].plot(xis, kappas_rI, "-.", label="reinterpolated")

    ax[2, 1].plot(xis, kappas_quat_ref, label="kappa quaternion current")
    ax[2, 1].plot(xis, kappas_lagr_ref, "--", label="kappa Lagrange current")
    ax[2, 1].plot(spring.qp, kappas_qp_ref, "x", label="qpi")
    ax[2, 1].plot(xis, kappas_rI_ref, "-.", label="reinterpolated")

    ax[3, 1].plot(xis, kappas_quat - kappas_quat_ref, label="kappa quaternion diff")
    ax[3, 1].plot(xis, kappas_lagr - kappas_lagr_ref, "--", label="kappa Lagrange diff")
    ax[3, 1].plot(spring.qp, kappas_qp - kappas_qp_ref, "x", label="qpi")
    ax[3, 1].plot(xis, kappas_rI - kappas_rI_ref, "-.", label="reinterpolated")

    # add grid and legend
    [[(axii.grid(), axii.legend()) for axii in axi] for axi in ax]

    plt.show()


if __name__ == "__main__":
    # G = 3200.0
    # rho = 1.0
    # L = 0.1
    # angles = np.linspace(-2 * np.pi, 2 * np.pi, 3)
    # omegas = np.zeros_like(angles)
    # for i, angle in enumerate(angles):
    #     omegas[i] = cardillo_linearize(0.0, 0.0, angle, G=G, rho=rho, L=L)

    # omega0_analytic = (2 * 0 + 1) / 2 * np.pi / L * np.sqrt(G / rho)

    # fig, ax = plt.subplots(1, 1)
    # ax = np.array([[ax]])
    # ax[0, 0].plot(angles, omegas, label="numerical")
    # ax[0, 0].plot(
    #     [angles[0], angles[-1]], [omega0_analytic] * 2, "--", label="analytical"
    # )

    # [[(axii.grid(), axii.legend()) for axii in axi] for axi in ax]
    # plt.show()

    # exit()

    polynomial_degree = 2

    nquadrature = polynomial_degree  # reduced integration
    # nquadrature = 2 * polynomial_degree * 2

    spring = TorsionSpring(
        GI=1.0, nquadrature=nquadrature, polynomial_degree=polynomial_degree
    )

    # without twist
    q = np.array([1.0, 0.0, 1.0, 0.0])

    # twist by an angle
    angle = np.pi / 4

    q = spring.twisted_configuration(
        *[i * angle / polynomial_degree for i in range(polynomial_degree + 1)]
    )
    q = spring.step_callback(0, q, 0)[0]
    print(f"q: {q}")
    give_me_results(spring, q)
