import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.discrete.discrete_export_base import make_glTF, make_glTF_modes
from cardillo.solver import Eigenmodes


class MassSpring:
    def __init__(self, m1, m2, k):
        self.m1 = m1
        self.m2 = m2
        self.k = k

        self.nq = self.nu = 2
        self.nla_g = 1

        self.q0 = np.array([0.0, 0.0])
        self.u0 = np.array([0.0, 0.0])

        self.y = 0.0
        self.z = 0.0

    # trivial kinematics
    def q_dot(self, t, q, u):
        return u

    def q_dot_u(self, t, q):
        return np.eye(2)

    # mass
    def M(self, t, q):
        return np.array([[self.m1, 0], [0, self.m2]])

    # spring
    def h(self, t, q, u):
        return np.array([-self.k * q[0], 0])

    def h_q(self, t, q, u):
        return np.array([[-self.k, 0], [0, 0]])

    def KN_h(self, t, q, u):
        return -self.h_q(t, q, u), np.zeros((2, 2))

    # constraint
    def g(self, t, q):
        return np.array([q[1] - q[0]])

    def g_q(self, t, q):
        return self.W_g(t, q).T

    def g_dot(self, t, q, u):
        return self.g_q(t, q) @ u

    def g_dot_u(self, t, q):
        return self.g_q(t, q)

    def g_dot_q(self, t, q, u):
        return 0.0

    def g_ddot(self, t, q, u, u_dot):
        return self.g_q(t, q) @ u_dot

    def W_g(self, t, q):
        return np.array([[-1.0], [1.0]])

    def Wla_g_q(self, t, q, la_g):
        return np.array([[0.0, 0.0], [0.0, 0.0]])

    def KN_g(self, t, q, la_g):
        return np.zeros((2, 2)), np.zeros((2, 2))

    # def g_q_T_mu_q(self, t, q, mu_g):
    #     return ...

    # visualization
    def _export_nodes(self, solution, idx):
        r_OP = np.zeros((len(solution.t), 3))
        r_OP[:, 0] = solution.q[:, self.qDOF[idx]] + 0.1 * idx
        r_OP[:, 1] = self.y
        r_OP[:, 2] = self.z
        v_P = np.zeros((len(solution.t), 3))
        v_P[:, 0] = solution.u[:, self.uDOF[idx]]
        return r_OP, v_P, None, None

    def _export_nodes_modes(self, solution, idx):
        r_OP = np.zeros(3)
        r_OP[0] = solution.q[self.qDOF[idx]] + 0.1 * idx
        r_OP[1] = self.y
        r_OP[2] = self.z

        Delta_z = solution.Delta_z[self.uDOF[idx]]
        Delta_r = np.zeros((3, Delta_z.shape[0]))
        Delta_r[0] = Delta_z
        return r_OP, Delta_r.T, None, None

    def export_blender(self, path, solution):
        for i in range(2):
            r_OP, v_P, P_IB, B_Omega = self._export_nodes(solution, i)
            make_glTF(path, f"{self.name}_{i}", solution.t, r_OP, v_P, P_IB, B_Omega)

    def export_blender_modes(self, path, solution):
        for i in range(2):
            r_OP, Delta_r, P_IB, B_Delta_phi = self._export_nodes_modes(solution, i)
            make_glTF_modes(
                path,
                f"{self.name}_{i}",
                solution.omegas,
                r_OP,
                Delta_r,
                P_IB,
                B_Delta_phi,
            )


if __name__ == "__main__":
    m1 = 1.0
    m2 = 1.0
    k = 1.0

    omega_ref = np.sqrt(k / (m1 + m2))

    system = System()
    mass_spring1 = MassSpring(m1 * 0.0, m2 * 1.0, k)
    mass_spring2 = MassSpring(m1 * 1.5, m2, 0.0)
    mass_spring3 = MassSpring(m1 * 3.5, m2 + 2.5, 0.0)
    system.add(mass_spring1)
    system.add(mass_spring2)
    system.add(mass_spring3)
    system.assemble()

    mass_spring1.z = 1.0
    mass_spring2.z = 2.0
    mass_spring3.z = 3.0

    solver_modes = Eigenmodes(system, system.sol0)

    sol_cheap = solver_modes.solve_cheap()
    sol_NS = solver_modes.solve()

    ################
    # blender export
    ################
    dir_name = Path(__file__).parent
    system.export_blender(dir_name, "blend_NS", sol_NS, create_blend=True)

    print(f"cheap            : {sol_cheap.omegas}")
    print(f"Nullspace        : {sol_NS.omegas}")
