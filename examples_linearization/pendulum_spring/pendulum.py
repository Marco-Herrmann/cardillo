import numpy as np
import matplotlib.pyplot as plt


from cardillo import System
from cardillo.math.approx_fprime import approx_fprime
from cardillo.solver import Rattle, BackwardEuler


def cubic_spring(k1, k3):
    def force(x):
        return k1 * x + k3 * x**3

    def force_x(x):
        return k1 + 3 * k3 * x**2

    return force, force_x


class Pendulum:
    def __init__(self, L, m, k1, k3, d):
        self.L = L
        self.m = m

        self.k1 = k1
        self.k3 = k3

        self.d = d

        spring = cubic_spring(k1, k3)
        self.spring = spring[0]
        self.spring_y = spring[1]

    def q_dot(self, t, q, u):
        return u

    def q_dot_u(self, t, q):
        return np.eye(self.nq, self.nu)

    def M(self, t, q):
        return self._M


class Pendulum_phi(Pendulum):
    def __init__(self, L, m, k1, k3, d, phi0=None, phi_dot0=None):
        super().__init__(L, m, k1, k3, d)
        self._M = m * L**2

        self.q0 = np.array([0.0 if phi0 is None else phi0])
        self.u0 = np.array([0.0 if phi_dot0 is None else phi_dot0])

        self.nq = self.nu = 1

    def h(self, t, q, u):
        phi = q[0]
        phi_dot = u[0]
        y = self.L * np.sin(phi)
        y_dot = self.L * phi_dot * np.cos(phi)
        F = self.spring(y) + self.d * y_dot
        return np.array([-self.L * np.cos(phi) * F])

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

    def r_OP(self, q):
        phi = q[0]
        return self.L * np.array([np.cos(phi), np.sin(phi)])

    def v_P(self, q, u):
        phi = q[0]
        phi_dot = u[0]
        return self.L * phi_dot * np.array([-np.sin(phi), np.cos(phi)])


class Pendulum_xy(Pendulum):
    def __init__(self, L, m, k1, k3, d, phi0=None, phi_dot0=None):
        super().__init__(L, m, k1, k3, d)
        self._M = m * np.eye(2, dtype=float)

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

    # spring stiffnesses
    k1 = -1.0
    k3 = 2.0

    # k1 = 1.0
    # k3 = 0.0

    # damper
    d = 0.1

    # show spring stiffness
    if False:
        spring = cubic_spring(k1, k3)
        xs = np.linspace(-L, L, 200)
        Fs = [spring[0](xi) for xi in xs]
        fig, ax = plt.subplots()
        ax.plot(xs, Fs)
        ax.grid()

    # create pendulum
    phi0 = np.pi / 3
    # phi0 = 0.0

    pendulums = [
        Pendulum_phi(L, m, k1, k3, d, phi0),
        Pendulum_xy(L, m, k1, k3, d, phi0),
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
    for iP in range(2):
        pendulum = pendulums[iP]
        qDOF = pendulum.qDOF
        uDOF = pendulum.uDOF
        ax[0, iP].plot(t, [pendulum.r_OP(qi[qDOF]) for qi in q])
        ax[1, iP].plot(t, [pendulum.v_P(qi[qDOF], ui[uDOF]) for qi, ui in zip(q, u)])

    plt.show()
