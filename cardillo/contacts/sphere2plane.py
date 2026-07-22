import numpy as np
from vtk import VTK_LINE


from cardillo.constraints._base import (
    concatenate_qDOF,
    concatenate_uDOF,
    auxiliary_functions,
)
from cardillo.math.algebra import ax2skew, cross3
from cardillo.math.approx_fprime import approx_fprime
from cardillo.math.prox import Sphere


# TODO: We have to add a function that computes the correct contact forces by
# application of A @ la_F. That should be done on system level and the solver
# calls this before the converged la_F's are stored.
# TODO: add orientation excitation of frame
class Sphere2Plane:
    def __init__(
        self,
        subsystem1,
        subsystem2,
        mu,
        radius,
        B_r_CP1=np.zeros(3),
        B_r_CP2=np.zeros(3),
        A_B1P=np.eye(3),
        e_N=None,
        e_F=None,
        xi1=None,
        xi2=None,
        anisotropy=np.ones(2),
        name="sphere_to_plane_contact",
    ):
        """Contact between a sphere and a plane modelled as unilateral constraint with set-valued Coulomb friction.

        Parameters
        ----------
        subsystem1 : object
            Subsystem that defines the plane.
            e_z-axis of P-basis is plane's normal direction: A_IP = A_IB1 @ A_B1P.
            P1 is point on plane: r_OP1 = r_OC1 + A_IB1 @ B_r_P1
        subsystem2 : object
            Subsystem containing the point P2 around which the spherical contact surface is defined.
            r_OP2 = r_OC2 + A_IB2 @ B_r_P2
        mu : float
            Frictional coefficient
        r : float
            Radius of spherical contact surface. Possible values are in [0, inf].
        B_r_CP1: np.ndarray (3,)
            Vector from center of mass of subsystem1 to point P1 in the plane in body-fixed coordinates of subsystem1.
        B_r_CP2: np.ndarray (3,)
            Vector from center of mass of subsystem2 to point P2 in the plane in body-fixed coordinates of subsystem2.
        e_N : float
            Restitution coefficient for Newton-like impact law in normal direction.
        e_N : float
            Restitution coefficient for Newton-like impact law for friction.
        xi1 : TODO
        xi2 : TODO
        anisotropy : np.ndarray (2,)
            Scaling factors for stretching the friction force reservoir in e_x and e_y-direction of the 'frame'.
            anisotropy=(1,1) corresponds to a circular force reservoir, i.e., isotropic Coulomb friction.
        name : str
            Name of contribution.
        """
        self.subsystem1 = subsystem1
        self.subsystem2 = subsystem2
        self.xi1 = xi1
        self.xi2 = xi2
        self.B_r_CP1 = B_r_CP1
        self.B_r_CP2 = B_r_CP2
        self.A_B1P = A_B1P

        self.radius = radius
        self.name = name

        self.nla_N = 1
        self.e_N = np.zeros(self.nla_N) if e_N is None else e_N * np.ones(self.nla_N)

        if mu > 0:
            raise NotImplementedError
            self.A = np.diag(anisotropy)
            self.nla_F = 2 * self.nla_N
            self.gamma_F = self.__gamma_F
            self.gamma_F_q = self.__gamma_F_q
            self.e_F = (
                np.zeros(self.nla_F) if e_F is None else e_F * np.ones(self.nla_F)
            )

            # fmt: off
            self.friction_laws = [
                ([0], [0, 1], Sphere(mu)), # Coulomb
            ]
            # fmt: on

    def assembler_callback(self):
        concatenate_qDOF(self)
        concatenate_uDOF(self)
        auxiliary_functions(self, self.B_r_CP1, self.B_r_CP2, self.A_B1P, None)

    ################
    # normal contact
    ################
    def g_N(self, t, q):
        n = self.A_IJ1(t, q)[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)

        return np.array([n @ (r_OJ2 - r_OJ1)]) - self.radius

    def g_N_q(self, t, q):
        n = self.A_IJ1(t, q)[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)

        n_q1 = self.A_IJ1_q1(t, q)[:, 2, :]
        r_OJ1_q1 = self.r_OJ1_q1(t, q)
        r_OJ2_q2 = self.r_OJ2_q2(t, q)

        nq1 = self._nq1
        g_N_q = np.zeros([self.nla_N, self._nq], dtype=q.dtype)
        g_N_q[:, :nq1] = (r_OJ2 - r_OJ1) @ n_q1 - n @ r_OJ1_q1
        g_N_q[:, nq1:] = n @ r_OJ2_q2

        return g_N_q

    def g_N_dot(self, t, q, u):
        n = self.A_IJ1(t, q)[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)

        n_dot = cross3(self.Omega1(t, q, u), n)
        v_J1 = self.v_J1(t, q, u)
        v_J2 = self.v_J2(t, q, u)

        return np.array(
            [n @ (v_J2 - v_J1) + n_dot @ (r_OJ2 - r_OJ1)],
            dtype=np.common_type(q, u),
        )

    def g_N_dot_q(self, t, q, u):
        n = self.A_IJ1(t, q)[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)

        Omega1 = self.Omega1(t, q, u)
        n_dot = cross3(Omega1, n)
        v_J1 = self.v_J1(t, q, u)
        v_J2 = self.v_J2(t, q, u)

        n_q1 = self.A_IJ1_q1(t, q)[:, 2, :]
        r_OJ1_q1 = self.r_OJ1_q1(t, q)
        r_OJ2_q2 = self.r_OJ2_q2(t, q)

        Omega1_q1 = self.Omega1_q1(t, q, u)
        n_dot_q1 = ax2skew(Omega1) @ n_q1 - ax2skew(n) @ Omega1_q1
        v_J1_q1 = self.v_J1_q1(t, q, u)
        v_J2_q2 = self.v_J2_q2(t, q, u)

        nq1 = self._nq1
        g_N_dot_q = np.zeros([self.nla_N, self._nq], dtype=q.dtype)
        g_N_dot_q[:, :nq1] = (
            (v_J2 - v_J1) @ n_q1
            - n @ v_J1_q1
            + (r_OJ2 - r_OJ1) @ n_dot_q1
            - n_dot @ r_OJ1_q1
        )
        g_N_dot_q[:, nq1:] = n @ v_J2_q2 + n_dot @ r_OJ2_q2
        return g_N_dot_q

    def g_N_dot_u(self, t, q):
        return self.W_N(t, q).T

    def W_N(self, t, q):
        n = self.A_IJ1(t, q)[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)

        J_n = ax2skew(-n) @ self.J_R1(t, q)
        J_J1 = self.J_J1(t, q)
        J_J2 = self.J_J2(t, q)

        nu1 = self._nu1
        W_N = np.zeros([self._nu, self.nla_N], dtype=q.dtype)
        W_N[:nu1, 0] = (r_OJ2 - r_OJ1) @ J_n - n @ J_J1
        W_N[nu1:, 0] = n @ J_J2

        return W_N

    def g_N_ddot(self, t, q, u, u_dot):
        n = self.A_IJ1(t, q)[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)

        Omega1 = self.Omega1(t, q, u)
        n_dot = cross3(Omega1, n)
        v_J1 = self.v_J1(t, q, u)
        v_J2 = self.v_J2(t, q, u)

        Psi1 = self.Psi1(t, q, u, u_dot)
        n_ddot = cross3(Psi1, n) + cross3(Omega1, n_dot)
        a_J1 = self.a_J1(t, q, u, u_dot)
        a_J2 = self.a_J2(t, q, u, u_dot)

        return np.array(
            [n @ (a_J2 - a_J1) + 2 * n_dot @ (v_J2 - v_J1) + n_ddot @ (r_OJ2 - r_OJ1)],
            dtype=np.common_type(q, u, u_dot),
        )

    def Wla_N_q(self, t, q, la_N):
        n = self.A_IJ1(t, q)[:, 2]
        n_tilde = ax2skew(n)
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)

        J_R1 = self.J_R1(t, q)
        J_n = -n_tilde @ J_R1
        J_J1 = self.J_J1(t, q)
        J_J2 = self.J_J2(t, q)

        n_q1 = self.A_IJ1_q1(t, q)[:, 2, :]
        r_OJ1_q1 = self.r_OJ1_q1(t, q)
        r_OJ2_q2 = self.r_OJ2_q2(t, q)

        J_R1_q1 = self.J_R1_q1(t, q)
        J_n_q1 = np.einsum("ijk,jl->kli", ax2skew(n_q1.T), J_R1) + np.einsum(
            "ji,jkl->ikl", n_tilde, J_R1_q1
        )  # minus sign in transpose/einsum
        J_J1_q1 = self.J_J1_q1(t, q)
        J_J2_q2 = self.J_J2_q2(t, q)

        nq1 = self._nq1
        nu1 = self._nu1
        w0_q = np.zeros([self._nu, self._nq], dtype=q.dtype)
        w0_q[:nu1, :nq1] = (
            -J_n.T @ r_OJ1_q1
            + np.einsum("i,ijk->jk", r_OJ2 - r_OJ1, J_n_q1)
            - J_J1.T @ n_q1
            - np.einsum("i,ijk->jk", n, J_J1_q1)
        )
        w0_q[:nu1, nq1:] = J_n.T @ r_OJ2_q2
        w0_q[nu1:, :nq1] = J_J2.T @ n_q1
        w0_q[nu1:, nq1:] = np.einsum("i,ijk->jk", n, J_J2_q2)

        Wla_N_q = w0_q * la_N[0]
        return Wla_N_q

    def KN_N(self, t, q, la_N):
        n = self.A_IJ1(t, q)[:, 2]
        n_tilde = ax2skew(n)
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)

        J_R1 = self.J_R1(t, q)
        J_n = -n_tilde @ J_R1
        J_J1 = self.J_J1(t, q)
        J_J2 = self.J_J2(t, q)

        J2_J1 = self.J2_J1(t, q)
        J2_J2 = self.J2_J2(t, q)

        v_tilde = ax2skew(r_OJ2 - r_OJ1)
        double_tilde = v_tilde @ n_tilde + n_tilde @ v_tilde

        nu1 = self._nu1
        W_N = np.zeros([self._nu, self.nla_N], dtype=q.dtype)
        W_N[:nu1, 0] = (r_OJ2 - r_OJ1) @ J_n - n @ J_J1
        W_N[nu1:, 0] = n @ J_J2

        w02 = np.zeros([self._nu, self._nu], dtype=q.dtype)
        w02[:nu1, :nu1] = (
            -J_n.T @ J_J1
            + 1 / 2 * J_R1.T @ double_tilde @ J_R1
            - J_J1.T @ J_n
            - np.einsum("i,ijk->jk", n, J2_J1)
        )
        w02[:nu1, nu1:] = J_n.T @ J_J2
        w02[nu1:, :nu1] = J_J2.T @ J_n
        w02[nu1:, nu1:] = np.einsum("i,ijk->jk", n, J2_J2)

        K = -w02 * la_N[0]
        N = np.zeros_like(K)
        return K, N

        g_N_qq = approx_fprime(
            q, lambda q_: approx_fprime(q_, lambda q__: self.g_N(t, q__))
        )
        B = np.block(
            [
                [self.subsystem1.q_dot_u(t, q[: self._nq1]), np.zeros((7, 6))],
                [np.zeros((7, 6)), self.subsystem2.q_dot_u(t, q[self._nq1 :])],
            ]
        )
        w02_num = B.T @ g_N_qq @ B
        K_num = -la_N[0] * w02_num

    ##########
    # friction
    ##########
    def __gamma_F(self, t, q, u):
        r_PS = -self.r * self.n(t)
        v_S = self.v_P(t, q, u) + cross3(self.Omega(t, q, u), r_PS)
        r_QS = self.r_OP(t, q) + r_PS - self.r_OQ(t)
        v_F = self.v_Q(t) + self.Omega_F_tilde(t) @ r_QS
        return self.A.T @ self.t1t2(t) @ (v_S - v_F)

    def __gamma_F_q(self, t, q, u):
        # return approx_fprime(q, lambda q: self.gamma_F(t, q, u))
        v_S_q = self.v_P_q(t, q, u) + self.r * ax2skew(self.n(t)) @ self.Omega_q(
            t, q, u
        )
        v_F_q = self.Omega_F_tilde(t) @ self.r_OP_q(t, q)
        return self.A.T @ self.t1t2(t) @ (v_S_q - v_F_q)

    def gamma_F_dot(self, t, q, u, u_dot):
        r_PS = -self.r * self.n(t)
        r_PS_dot = -self.r * self.n_dot(t)
        v_S = self.v_P(t, q, u) + cross3(self.Omega(t, q, u), r_PS)
        a_S = (
            self.a_P(t, q, u, u_dot)
            + cross3(self.Psi(t, q, u, u_dot), r_PS)
            + cross3(self.Omega(t, q, u), r_PS_dot)
        )
        r_QS = self.r_OP(t, q) + r_PS - self.r_OQ(t)
        r_QS_dot = self.v_P(t, q, u) + r_PS_dot - self.v_Q(t)
        v_F = self.v_Q(t) + self.Omega_F_tilde(t) @ r_QS
        a_F = (
            self.a_Q(t) + self.Psi_F_tilde(t) @ r_QS + self.Omega_F_tilde(t) @ r_QS_dot
        )
        return self.A.T @ (self.t1t2(t) @ (a_S - a_F) + self.t1t2_dot(t) @ (v_S - v_F))

    def gamma_F_dot_q(self, t, q, u, u_dot):
        # return approx_fprime(q, lambda q: self.gamma_F_dot(t, q, u, u_dot))
        r_PS_tilde = ax2skew(-self.r * self.n(t))
        r_PS_dot_tilde = ax2skew(-self.r * self.n_dot(t))
        v_S_q = self.v_P_q(t, q, u) - r_PS_tilde @ self.Omega_q(t, q, u)
        a_S_q = (
            self.a_P_q(t, q, u, u_dot)
            - r_PS_tilde @ self.Psi_q(t, q, u, u_dot)
            - r_PS_dot_tilde @ self.Omega_q(t, q, u)
        )

        v_F_q = self.Omega_F_tilde(t) @ self.r_OP_q(t, q)
        a_F_q = self.Psi_F_tilde(t) @ self.r_OP_q(t, q) + self.Omega_F_tilde(
            t
        ) @ self.v_P_q(t, q, u)

        return self.A.T @ (
            self.t1t2(t) @ (a_S_q - a_F_q) + self.t1t2_dot(t) @ (v_S_q - v_F_q)
        )

    def gamma_F_dot_u(self, t, q, u, u_dot):
        # return approx_fprime(u, lambda u: self.gamma_F_dot(t, q, u, u_dot))
        r_PS_tilde = ax2skew(-self.r * self.n(t))
        a_S_u = self.a_P_u(t, q, u, u_dot) - r_PS_tilde @ self.Psi_u(t, q, u, u_dot)
        J_P = self.J_P(t, q)
        a_F_u = self.Omega_F_tilde(t) @ J_P
        J_S = self.J_P - r_PS_tilde @ self.J_R(t, q)
        return self.A.T @ (self.t1t2(t) @ (a_S_u - a_F_u) + self.t1t2_dot(t) @ J_S)

    def gamma_F_u(self, t, q):
        # return approx_fprime(np.zeros(self.nu), lambda u: self.gamma_F(t, q, u))
        r_PS_tilde = ax2skew(-self.r * self.n(t))
        J_S = self.J_P(t, q) - r_PS_tilde @ self.J_R(t, q)
        return self.A.T @ self.t1t2(t) @ J_S

    def W_F(self, t, q):
        return self.gamma_F_u(t, q).T

    def Wla_F_q(self, t, q, la_F):
        J_S_q = self.J_P_q(t, q) + self.r * np.einsum(
            "ij,jkl->ikl", ax2skew(self.n(t)), self.J_R_q(t, q)
        )
        Wla_F_q = np.einsum("i,ij,jkl->kl", la_F, self.A.T @ self.t1t2(t), J_S_q)
        return Wla_F_q
        # Wla_F_q_num = approx_fprime(q, lambda q: self.gamma_F_u(t, q).T @ la_F)
        # diff = Wla_F_q - Wla_F_q_num
        # error = np.linalg.norm(diff)
        # print(f"error Wla_F_q: {error}")
        # return Wla_F_q_num

    ############
    # vtk export
    ############
    def export(self, sol_i, **kwargs):
        r_OP = self.r_OP(sol_i.t, sol_i.q[self.qDOF])
        n = self.n(sol_i.t)
        t1, t2 = self.t1t2(sol_i.t)
        g_N = self.g_N(sol_i.t, sol_i.q[self.qDOF])
        P_N = sol_i.P_N[self.la_NDOF]
        r_PC1 = -self.r * n
        r_QC2 = r_OP - self.r_OQ(sol_i.t) - n * (g_N + self.r)
        points = [r_OP + r_PC1, r_OP - n * (g_N + self.r)]
        cells = [(VTK_LINE, [0, 1])]
        A_IB1 = self.A_IB(sol_i.t, sol_i.q[self.qDOF])
        A_IB2 = self.frame.A_IB(sol_i.t)
        point_data = dict(
            v_Ci=[
                self.subsystem.v_P(
                    sol_i.t,
                    sol_i.q[self.qDOF],
                    sol_i.u[self.uDOF],
                    self.xi,
                    A_IB1.T @ r_PC1,
                ),
                self.frame.v_P(sol_i.t, B_r_CP=A_IB2.T @ r_QC2),
            ],
            Omega=[
                self.Omega(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF]),
                A_IB2 @ self.frame.B_Omega(sol_i.t),
            ],
            n=[-n, n],
            t1=[-t1, t1],
            t2=[-t2, t2],
            P_N=[P_N, P_N],
        )
        cell_data = dict(
            g_N=[g_N],
            g_N_dot=[self.g_N_dot(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])],
        )

        if hasattr(self, f"gamma_F"):
            cell_data["gamma_F"] = [
                self.gamma_F(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])
            ]
            P_F = sol_i.P_F[self.la_FDOF]
            point_data["P_F"] = np.array([P_F, P_F])

        return points, cells, point_data, cell_data

    def export_blender(self, path, solution):
        from warnings import warn

        warn("Sphere2Plane.export_blender not implemented")

    def export_blender_modes(self, path, solution):
        from warnings import warn

        warn("Sphere2Plane.export_blender_modes not implemented")
