import matplotlib.pyplot as plt
import numpy as np


from cardillo.math.rotations import (
    A_IB_basic,
    Exp_SO3_quat,
    Log_SO3_quat,
    T_SO3_quat,
    T_SO3_quat_P,
    T_SO3_inv_quat,
)
from cardillo.rods.discretization import gauss


class TorsionSpring:
    def __init__(self, *, GI=1.0, L=1.0, normalize=True, nquadrature=1):
        self.GI = GI
        self.L = L

        self.nq = 4
        self.nu = 2

        self.q0 = np.array([1.0, 0.0, 1.0, 0.0], dtype=float)
        self.u0 = np.array([0.0, 0.0], dtype=float)

        self.nla_c = 1
        self.la_c0 = np.array([0.0])

        self.nquadrature = nquadrature
        self.qp, self.qw = gauss(self.nquadrature, [0, 1])

        # normalize in B
        self.normalize = normalize

    def Nq(self, xi):
        Nq = np.zeros((4, 4), dtype=float)
        Nq[0, 0] = 1 - xi
        Nq[1, 1] = 1 - xi
        Nq[0, 2] = xi
        Nq[1, 3] = xi
        return Nq

    def Nq_xi(self, xi):
        Nq_xi = np.zeros((4, 4), dtype=float)
        Nq_xi[0, 0] = -1
        Nq_xi[1, 1] = -1
        Nq_xi[0, 2] = 1
        Nq_xi[1, 3] = 1
        return Nq_xi

    def Nu(self, xi):
        Nu = np.zeros((3, 2), dtype=float)
        Nu[0, 0] = 1 - xi
        Nu[0, 1] = xi
        return Nu

    def Nu_xi(self, xi):
        Nu_xi = np.zeros((1, 2), dtype=float)
        Nu_xi[0, 0] = -1
        Nu_xi[0, 1] = 1
        return Nu_xi

    def kappa(self, xi, q):
        Pi = self.Nq(xi) @ q
        Pi_xi = self.Nq_xi(xi) @ q

        B_kappa_IB_bar = T_SO3_quat(Pi, normalize=True) @ Pi_xi
        return B_kappa_IB_bar[0] / self.L

    def q_dot_u(self, t, q):
        P0 = self.Nq(0.0) @ q
        P1 = self.Nq(1.0) @ q
        q_dot_u = np.zeros((self.nq, self.nu), dtype=float)
        q_dot_u[:2, 0] = T_SO3_inv_quat(P0, normalize=self.normalize)[:2, 0]
        q_dot_u[2:, 1] = T_SO3_inv_quat(P1, normalize=self.normalize)[:2, 0]
        return q_dot_u

    def W_c(self, t, q):
        W_c = np.zeros((self.nu, self.nla_c), dtype=float)

        for i in range(self.nquadrature):
            qpi = self.qp[i]
            wpi = self.qw[i]

            Nu_xi = self.Nu_xi(qpi)
            W_c -= Nu_xi.T * wpi

        return W_c

    def K_c_inv(self):
        K_c_inv = np.zeros((self.nla_c, self.nla_c), dtype=float)
        K_c_inv[0, 0] = self.L / self.GI

        return K_c_inv

    def l_c_master(self, q):
        l_c = np.zeros(self.nla_c, dtype=float)
        l_c_q = np.zeros((self.nla_c, self.nq), dtype=float)

        for i in range(self.nquadrature):
            qpi = self.qp[i]
            wpi = self.qw[i]

            Pi = self.Nq(qpi) @ q
            Pi_xi = self.Nq_xi(qpi) @ q

            B_kappa_IB_bar = T_SO3_quat(Pi, normalize=True) @ Pi_xi
            B_kappa_IB_bar_q = T_SO3_quat(Pi, normalize=True) @ self.Nq_xi(
                qpi
            ) + np.einsum(
                "ijk, j -> ik", T_SO3_quat_P(Pi, normalize=True) @ self.Nq(qpi), Pi_xi
            )

            kappa_x = B_kappa_IB_bar[0]
            kappa_x_q = B_kappa_IB_bar_q[0]

            l_c += kappa_x * wpi
            l_c_q += kappa_x_q * wpi

        return l_c, l_c_q

    def c(self, t, q, u, la_c):
        return self.K_c_inv() @ la_c - self.l_c_master(q)[0]

    def c_q(self, t, q):
        return -self.l_c_master(q)[1]

    def normalize_q(self, q):
        for i in range(2):
            DOF = range(2 * i, 2 * i + 2)
            q[DOF] /= np.sqrt(q[DOF] @ q[DOF])

        return q

    @staticmethod
    def twisted_configuration(alpha0, alpha1):
        return np.concatenate(
            [Log_SO3_quat(A_IB_basic(alpha).x)[:2] for alpha in [alpha0, alpha1]]
        )

    def angle(self, xi, q):
        A_IB = Exp_SO3_quat(self.Nq(xi) @ q)
        ca = A_IB[1, 1]
        sa = A_IB[2, 1]
        return np.atan2(sa, ca)


if __name__ == "__main__":
    nquadrature = 1  # reduced integration
    nquadrature = 2
    spring = TorsionSpring(GI=1.0, nquadrature=nquadrature)
    t = 0.0

    # without twist
    q = np.array([1.0, 0.0, 1.0, 0.0])

    # twist by an angle
    angle = np.pi / 4

    q = spring.twisted_configuration(0.0, angle)

    # make unit quaternions
    q = spring.normalize_q(q)
    print(f"q: {q}")

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
    P_norm = np.array([Pi / (Pi @ Pi) for Pi in P])
    alphas = np.array([spring.angle(xi, q) for xi in xis])
    kappas = np.array([spring.kappa(xi, q) for xi in xis])

    kappa_quadratisch = (
        lambda xi: 2 * spring.kappa(0.0, q) * (xi - 0.5) * (xi - 1.0)
        - 4 * spring.kappa(0.5, q) * (xi - 0.0) * (xi - 1.0)
        + 2 * spring.kappa(1.0, q) * (xi - 0.0) * (xi - 0.5)
    )

    fig, ax = plt.subplots(2, 2)
    # plot quaternion values
    ax[0, 0].plot(xis, P[:, 0], label="p0")
    ax[0, 0].plot(xis, P[:, 1], label="p1")
    ax[1, 0].plot(xis, P_norm[:, 0], label="p0 / ||P||^2")
    ax[1, 0].plot(xis, P_norm[:, 1], label="p1 / ||P||^2")

    # plot angle
    ax[0, 1].plot(xis, alphas, label="alpha quaternion")
    ax[0, 1].plot([0.0, 1.0], [0.0, angle], "--", label="alpha Lagrange")

    # plot torsion
    ax[1, 1].plot(xis, kappas, label="kappa quaternion")
    ax[1, 1].plot(
        [0.0, 1.0], [angle / spring.L, angle / spring.L], "--", label="kappa Lagrange"
    )
    ax[1, 1].plot(spring.qp, [angle] * spring.nquadrature, "x", label="qpi")

    ax[1, 1].plot(
        xis,
        kappa_quadratisch(xis),
        "--",
        label="quadratic approximation at {0.0, 0.5, 1.0}",
    )

    # add grid and legend
    [[(axii.grid(), axii.legend()) for axii in axi] for axi in ax]

    plt.show()
