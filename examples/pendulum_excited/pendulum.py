import numpy as np
import matplotlib.pyplot as plt


from cardillo import System
from cardillo.math.approx_fprime import approx_fprime
from cardillo.solver import Rattle, BackwardEuler


class Pendulum:
    def __init__(self, L, m, g, e, d):
        self.L = L
        self.m = m
        self.g = g
        self.e = e
        self.d = d

    def q_dot(self, t, q, u):
        return u

    def q_dot_u(self, t, q):
        return np.eye(self.nq, self.nu)

    def M(self, t, q):
        return self._M


class Pendulum_phi(Pendulum):
    def __init__(self, L, m, g, e, d, phi0=None, phi_dot0=None):
        super().__init__(L, m, g, e, d)
        self._M = m * L**2

        self.q0 = np.array([0.0 if phi0 is None else phi0])
        self.u0 = np.array([0.0 if phi_dot0 is None else phi_dot0])

        self.nq = self.nu = 1

    def h(self, t, q, u):
        phi = q[0]
        cphi = np.cos(phi)
        sphi = np.sin(phi)

        M_g = -self.L * self.m * self.g * sphi
        M_e = -self.L * self.m * self.e(t)[2] * cphi

        phi_dot = u[0]
        M_d = -phi_dot * self.d
        return np.array([M_g + M_e + M_d])

    def h_q(self, t, q, u):
        return approx_fprime(q, lambda q_: self.h(t, q_, u))
        phi = q[0]
        y = np.sin(phi)
        y_q = np.cos(phi)
        F_y = self.spring_y(y)
        return np.array([[-F_y * y_q]])

    def h_u(self, t, q, u):
        return approx_fprime(u, lambda u_: self.h(t, q, u_))
        return np.array([[-self.d]])

    def r_OP(self, t, q):
        phi = q[0]
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        return self.L * np.array([cphi, -sphi]) + np.array([self.e(t)[0], 0.0])

    def v_P(self, t, q, u):
        phi = q[0]
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        phi_dot = u[0]
        return self.L * phi_dot * np.array([-sphi, -cphi]) + np.array(
            [self.e(t)[1], 0.0]
        )


class Pendulum_xy(Pendulum):
    def __init__(self, L, m, k1, k3, d, phi0=None, phi_dot0=None):
        super().__init__(L, m, k1, k3, d)
        self._M = m * np.eye(2, dtype=float)

        from warnings import warn

        warn("damping is 0.0")
        self.d = 0.0

        phi0_ = 0.0 if phi0 is None else phi0
        phi_dot0_ = 0.0 if phi_dot0 is None else phi_dot0
        self.q0 = L * np.array([np.cos(phi0_), np.sin(phi0_)])
        self.u0 = L * phi_dot0_ * np.array([-np.sin(phi0_), np.cos(phi0_)])

        self.nq = self.nu = 2
        self.nla_g = 1

    def h(self, t, q, u):
        y = q[1]
        y_dot = u[1]
        F = self.spring(y) + self.d * y_dot
        return np.array([0.0, -F])

    def h_q(self, t, q, u):
        y = q[1]
        F_y = self.spring_y(y)
        return np.array([[0.0, 0.0], [0.0, -F_y]])

    def h_u(self, t, q, u):
        return np.array([[0.0, 0.0], [0, -self.d]])

    def r_OP(self, q):
        return q

    def v_P(self, q, u):
        return u

    def g(self, t, q):
        return q @ q - self.L**2

    def g_q(self, t, q):
        return 2.0 * q

    def g_dot(self, t, q, u):
        return 2.0 * q @ u

    def g_dot_q(self, t, q, u):
        return 2.0 * u

    def g_dot_u(self, t, q):
        return 2.0 * q

    def g_ddot(self, t, q, u, u_dot):
        return 2.0 * u @ u + 2.0 * q @ u_dot

    def W_g(self, t, q):
        return 2.0 * np.reshape(q, (2, 1))

    def Wla_g_q(self, t, q, la_g):
        return 2.0 * np.eye(2, dtype=float) * la_g


if __name__ == "__main__":
    # geometry and mass
    L = 1.0
    m = 1.0

    # gravity
    g = 9.81

    # excitation
    om = 1.0 * np.pi
    e_hat = L / 2
    e = lambda t: e_hat * np.array(
        [
            np.sin(om * t),
            om * np.cos(om * t),
            -(om**2) * np.sin(om * t),
        ]
    )

    print(f"{om = }, {np.sqrt(g / L) = }")

    # damper
    d = 0.2

    # create pendulum
    # phi0 = np.pi/3
    phi0 = 0.0

    pendulums = [
        Pendulum_phi(L, m, g, e, d, phi0),
        # Pendulum_xy(L, m, k1, k3, d, phi0)
    ]

    # assemble the system
    system = System()
    system.add(*pendulums)
    system.assemble()

    # simulation
    t1 = 50.0
    dt = 5e-2
    sol = Rattle(system, t1, dt).solve()
    # sol = BackwardEuler(system, t1, dt).solve()

    t = sol.t
    q = sol.q
    u = sol.u

    fig, ax = plt.subplots(2, 2)
    for iP in range(len(pendulums)):
        pendulum = pendulums[iP]
        qDOF = pendulum.qDOF
        uDOF = pendulum.uDOF
        ax[0, iP].plot(t, [pendulum.r_OP(ti, qi[qDOF]) for ti, qi in zip(t, q)])
        ax[1, iP].plot(
            t, [pendulum.v_P(ti, qi[qDOF], ui[uDOF]) for ti, qi, ui in zip(t, q, u)]
        )

        ax[0, iP].plot(t, [e(ti)[0] for ti in t])

    plt.show()
