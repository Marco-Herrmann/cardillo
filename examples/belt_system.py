import numpy as np
from scipy.linalg import solve
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cardillo.math.prox import Sphere
from cardillo import System
from cardillo.solver import (
    Moreau,
    BackwardEuler,
    SolverOptions,
    ScipyIVP,
)


class belt_system:
    def __init__(
        self, length, density, stiffness, damping, drive_roller_velocity, drive_roller_acceleration, F_friction, q0, u0, nelements=10, 
    ):
        
        self.stiffness = stiffness
        self.damping = damping
        self.density = density
        self.length = length
        self.nelement = nelements
        self.nnd = nelements + 1
        self.Delta_x = length / self.nelement

        # if drive_roller_velocity is not callable:
        #     self.drive_roller_velocity = lambda t: drive_roller_velocity
        # else:
        #     self.drive_roller_velocity = drive_roller_velocity

        self.drive_roller_velocity = drive_roller_velocity
        self.drive_roller_acceleration = drive_roller_acceleration

        

        self.elDOF = np.vstack((np.arange(self.nelement), np.arange(self.nelement)+1)).T

        self.A_e = self.Delta_x * np.array([[1/3, 1/6], [1/6, 1/3]])

        self.nq = self.nnd
        self.nu = self.nnd
        self.q0 = q0
        self.u0 = u0
        assert self.nq == len(q0)
        assert self.nu == len(u0)

        self.F_friction = F_friction

        # fmt: off
        self.friction_laws = [
            ([], [0], Sphere(F_friction)),
        ]
        # fmt: on

        # self.nla_gamma = 1

        # self.nla_F = self.nnd
        self.nla_F = 1
        self.e_F = np.zeros(self.nla_F)

        self.A = np.zeros((self.nq, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            self.A[elDOF[:, None], elDOF] += self.A_e

        self.A_inv = np.linalg.inv(self.A)

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        return self.q_dot_u(t, q) @ u
    
    def q_dot_u(self, t, q):
        E = np.zeros((self.nq, self.nu), dtype=np.common_type(q))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            qe = q[elDOF]
            Delta_s = qe[1] - qe[0]
            E[elDOF[:, None], elDOF] += Delta_s / self.Delta_x * self.A_e

        return -self.A_inv @ E

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        M = np.zeros((self.nu, self.nu), dtype=np.common_type(q))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            qe = q[elDOF]
            Delta_s = qe[1] - qe[0]
            M[elDOF[:, None], elDOF] += self.density * Delta_s / self.Delta_x * self.A_e

        return M

    def h(self, t, q, u):
        return self.f_elastic(q) + self.f_viscous(q, u) + self.f_gyr(q, u)

    def f_gyr(self, q, u):
        f_gyr = np.zeros(self.nu, dtype=np.common_type(q, u))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            qe = q[elDOF]
            Delta_s = qe[1] - qe[0]
            ue = u[elDOF]
            Delta_v = ue[1] - ue[0]
            f_gyr[elDOF] -= self.density * Delta_s * Delta_v / (self.Delta_x**2) * self.A_e @ ue
        return f_gyr
    
    def f_elastic(self, q):
        f_elastic = np.zeros(self.nu, dtype=np.common_type(q))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            qe = q[elDOF]
            Delta_s = qe[1] - qe[0]
            f_elastic[elDOF] -= self.stiffness * (self.Delta_x / Delta_s - 1) * np.array([-1, 1])
        return f_elastic
    
    def f_viscous(self, q, u):
        f_viscous = np.zeros(self.nu, dtype=np.common_type(q, u))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            qe = q[elDOF]
            Delta_s = qe[1] - qe[0]
            ue = u[elDOF]
            Delta_v = ue[1] - ue[0]
            f_viscous[elDOF] -= self.damping / (Delta_s) * np.array([-Delta_v, Delta_v])
        return f_viscous
    
    ######################
    # # velocity constraint
    # ######################
    # def gamma(self, t, q, u):
    #     return np.array([u[0]]) - self.drive_roller_velocity(t)

    # def gamma_u(self, t, q):
    #     gamma_u = np.zeros((self.nla_gamma, self.nu), dtype=q.dtype)
    #     gamma_u[0, 0] = 1
    #     return gamma_u

    # def gamma_dot(self, t, q, u, u_dot):
    #     return np.array([u_dot[0]]) - self.drive_roller_acceleration(t)

    # def W_gamma(self, t, q):
    #     return self.gamma_u(t, q).T

    # def Wla_gamma_q(self, t, q, la_F):
    #     return np.zeros((self.nu, self.nq))



    #########
    # friction
    # ##########
    def gamma_F(self, t, q, u):
        return np.array([u[0]])

    def gamma_F_u(self, t, q):
        gamma_F_u = np.zeros((self.nla_F, self.nu), dtype=q.dtype)
        gamma_F_u[0, 0] = 1
        return gamma_F_u

    def gamma_F_dot(self, t, q, u, u_dot):
        return np.array([u_dot[0]])

    def W_F(self, t, q):
        return self.gamma_F_u(t, q).T

    def Wla_F_q(self, t, q, la_F):
        return np.zeros((self.nu, self.nq))
    

    # ########
    # # friction
    # ##########
    # def gamma_F(self, t, q, u):
    #     return u

    # def gamma_F_u(self, t, q):
    #     return np.eye(self.nu)

    # def gamma_F_dot(self, t, q, u, u_dot):
    #     return u_dot

    # def W_F(self, t, q):
    #     return self.gamma_F_u(t, q).T

    # def Wla_F_q(self, t, q, la_F):
    #     return np.zeros((self.nu, self.nq))


if __name__ == "__main__":
    nelements = 2
    length = 1
    density = 600
    thickness = 2.3e-3
    stiffness = 7.e7 * thickness
    # damping = 1e3
    damping = 0
    F_friction = 1e3
    angular_frequency = 2 * 2 * np.pi
    drive_roller_velocity = lambda t: -0.2 * np.sin(angular_frequency * t)
    drive_roller_acceleration = lambda t: -0.2 * np.cos(angular_frequency * t) * angular_frequency


    q0 = np.linspace(0, length, nelements+1) * 0.98
    u0 = np.zeros(nelements+1)
    # q0 = np.linspace(0, length, nelements+1)
    # u0 = -np.ones(nelements+1)

    belt = belt_system(
        length=1,
        density=density,
        stiffness=stiffness,
        damping=damping,
        drive_roller_velocity=drive_roller_velocity,
        drive_roller_acceleration=drive_roller_acceleration,
        F_friction=F_friction, 
        q0=q0, 
        u0=u0, 
        nelements=nelements, 
        )

    system = System()
    system.add(belt)

    system.assemble()
    
    t_final = 1
    dt = 1e-3

    # solver = ScipyIVP(system, t_final, dt)
    # # solver, label = Moreau(system, t_final, dt), "Moreau"
    
    solver, label = (
        BackwardEuler(
            # system, t_final, dt, options=SolverOptions(reuse_lu_decomposition=True)
            system, t_final, dt, options=SolverOptions(reuse_lu_decomposition=True, numerical_jacobian_method="2-point")
        ),
        "BackwardEuler",
    )

    sol = solver.solve()
    t = sol.t
    q = sol.q
    u = sol.u

    # P_F = sol.P_F

    fig, ax = plt.subplots(1, 2)

    ax[0].set_title("x(t)")
    ax[0].set_title("x(t)")
    ax[0].plot(q[:, :], t, "-k")
    ax[0].grid()
    # ax[0].legend()

    plt.show()

    # ax[1].set_title("u_x(t)")
    # ax[1].plot(t, u[:, 0], "-k", label=label)
    # ax[1].legend()

    # ax[2].set_title("P_F(t)")
    # ax[2].plot(t, P_F[:, 0], "-k", label=label)
    # ax[2].legend()

    # plt.show()
