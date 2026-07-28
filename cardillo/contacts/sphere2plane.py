import numpy as np
from vtk import VTK_LINE


from cardillo.constraints._base import (
    concatenate_qDOF,
    concatenate_uDOF,
    auxiliary_functions,
)
from cardillo.discrete.discrete_export_base import make_glTF, make_glTF_arrow
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
            self.A = np.diag(anisotropy)
            self.nla_F = 2 * self.nla_N
            self.gamma_F = lambda t, q, u: self.A.T @ self._gamma(t, q, u)[:2]
            self.gamma_F_q = lambda t, q, u: self.A.T @ self._gamma_q(t, q, u)[:2]
            self.gamma_F_u = lambda t, q: self.A.T @ self._gamma_u(t, q)[:2]
            self.gamma_F_dot = (
                lambda t, q, u, u_dot: self.A.T @ self._gamma_dot(t, q, u, u_dot)[:2]
            )
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

    # cached methods for velocity level, acceleration level and generalized force direction
    # TODO: shall we make all helper function in J1-system?
    def _gamma(self, t, q, u):
        A_IJ1 = self.A_IJ1(t, q)
        n = A_IJ1[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)
        r_J1J2 = r_OJ2 - r_OJ1
        r_J1C1 = r_J1J2 - n * (n @ r_J1J2)
        r_J2C2 = -self.radius * n

        v_J1 = self.v_J1(t, q, u)
        v_J2 = self.v_J2(t, q, u)
        Omega1 = self.Omega1(t, q, u)
        Omega2 = self.Omega2(t, q, u)
        v_C1 = v_J1 + cross3(Omega1, r_J1C1)
        v_C2 = v_J2 + cross3(Omega2, r_J2C2)

        return A_IJ1.T @ (v_C2 - v_C1)

    def _gamma_q(self, t, q, u):
        A_IJ1 = self.A_IJ1(t, q)
        n = A_IJ1[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)
        r_J1J2 = r_OJ2 - r_OJ1
        r_J1C1 = r_J1J2 - n * (n @ r_J1J2)
        r_J2C2 = -self.radius * n

        v_J1 = self.v_J1(t, q, u)
        v_J2 = self.v_J2(t, q, u)
        Omega1 = self.Omega1(t, q, u)
        Omega2 = self.Omega2(t, q, u)
        v_C1 = v_J1 + cross3(Omega1, r_J1C1)
        v_C2 = v_J2 + cross3(Omega2, r_J2C2)

        # derivatives
        A_IJ1_q1 = self.A_IJ1_q1(t, q)
        n_q1 = A_IJ1_q1[:, 2, :]
        r_OJ1_q1 = self.r_OJ1_q1(t, q)
        r_OJ2_q2 = self.r_OJ2_q2(t, q)
        r_J1C1_q1 = (
            -r_OJ1_q1
            - (n @ r_J1J2) * n_q1
            - np.outer(n, r_J1J2 @ n_q1)
            + np.outer(n, n @ r_OJ1_q1)
        )
        r_J1C1_q2 = r_OJ2_q2 - np.outer(n, n @ r_OJ2_q2)
        r_J2C2_q1 = -self.radius * n_q1

        v_J1_q1 = self.v_J1_q1(t, q, u)
        v_J2_q2 = self.v_J2_q2(t, q, u)
        Omega1_q1 = self.Omega1_q1(t, q, u)
        Omega2_q2 = self.Omega2_q2(t, q, u)
        v_C1_q1 = v_J1_q1 - ax2skew(r_J1C1) @ Omega1_q1 + ax2skew(Omega1) @ r_J1C1_q1
        v_C1_q2 = ax2skew(Omega1) @ r_J1C1_q2
        v_C2_q1 = ax2skew(Omega2) @ r_J2C2_q1
        v_C2_q2 = v_J2_q2 - ax2skew(r_J2C2) @ Omega2_q2

        # compute
        nq1 = self._nq1
        gamma_q = np.zeros([3, self._nq], dtype=q.dtype)
        gamma_q[:, :nq1] = A_IJ1.T @ (v_C2_q1 - v_C1_q1) + np.einsum(
            "ijk,i->jk", A_IJ1_q1, v_C2 - v_C1
        )
        gamma_q[:, nq1:] = A_IJ1.T @ (v_C2_q2 - v_C1_q2)
        return gamma_q

    def _gamma_u(self, t, q):
        A_IJ1 = self.A_IJ1(t, q)
        n = A_IJ1[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)
        r_J1J2 = r_OJ2 - r_OJ1
        r_J1C1 = r_J1J2 - n * (n @ r_J1J2)
        r_J2C2 = -self.radius * n

        J_J1 = self.J_J1(t, q)
        J_J2 = self.J_J2(t, q)
        J_R1 = self.J_R1(t, q)
        J_R2 = self.J_R2(t, q)
        J_C1 = J_J1 - ax2skew(r_J1C1) @ J_R1
        J_C2 = J_J2 - ax2skew(r_J2C2) @ J_R2

        nu1 = self._nu1
        gamma_u = np.zeros([3, self._nu], dtype=q.dtype)
        gamma_u[:, :nu1] = -J_C1
        gamma_u[:, nu1:] = J_C2

        return A_IJ1.T @ gamma_u

    def _gamma_dot(self, t, q, u, u_dot):
        A_IJ1 = self.A_IJ1(t, q)
        n = A_IJ1[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)
        r_J1J2 = r_OJ2 - r_OJ1
        r_J1C1 = r_J1J2 - n * (n @ r_J1J2)
        r_J2C2 = -self.radius * n

        v_J1 = self.v_J1(t, q, u)
        v_J2 = self.v_J2(t, q, u)
        Omega1 = self.Omega1(t, q, u)
        Omega2 = self.Omega2(t, q, u)
        v_C1 = v_J1 + cross3(Omega1, r_J1C1)
        v_C2 = v_J2 + cross3(Omega2, r_J2C2)

        # time derivatives
        n_dot = cross3(Omega1, n)
        v_J1J2 = v_J2 - v_J1
        r_J1C1_dot = (
            v_J1J2 - n_dot * (n @ r_J1J2) - n * (n_dot @ r_J1J2) - n * (n @ v_J1J2)
        )
        r_J2C2_dot = -self.radius * n_dot

        a_J1 = self.a_J1(t, q, u, u_dot)
        a_J2 = self.a_J2(t, q, u, u_dot)
        Psi1 = self.Psi1(t, q, u, u_dot)
        Psi2 = self.Psi2(t, q, u, u_dot)
        v_C1_dot = a_J1 + cross3(Psi1, r_J1C1) + cross3(Omega1, r_J1C1_dot)
        v_C2_dot = a_J2 + cross3(Psi2, r_J2C2) + cross3(Omega2, r_J2C2_dot)

        # compute
        gamma_dot = A_IJ1.T @ (v_C2_dot - v_C1_dot - cross3(Omega1, v_C2 - v_C1))
        return gamma_dot

    def _Wla(self, t, q, J1_la):
        A_IJ1 = self.A_IJ1(t, q)
        n = A_IJ1[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)
        r_J1J2 = r_OJ2 - r_OJ1
        r_J1C1 = r_J1J2 - n * (n @ r_J1J2)
        r_J2C2 = -self.radius * n

        J_J1 = self.J_J1(t, q)
        J_J2 = self.J_J2(t, q)
        J_R1 = self.J_R1(t, q)
        J_R2 = self.J_R2(t, q)
        J_C1 = J_J1 - ax2skew(r_J1C1) @ J_R1
        J_C2 = J_J2 - ax2skew(r_J2C2) @ J_R2

        F = A_IJ1 @ J1_la
        nu1 = self._nu1
        Wla = np.zeros([self._nu], dtype=q.dtype)
        Wla[:nu1] = -F @ J_C1
        Wla[nu1:] = F @ J_C2

        return Wla

    def _Wla_q(self, t, q, J1_F):
        A_IJ1 = self.A_IJ1(t, q)
        n = A_IJ1[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)
        r_J1J2 = r_OJ2 - r_OJ1
        r_J1C1 = r_J1J2 - n * (n @ r_J1J2)
        r_J2C2 = -self.radius * n

        J_J1 = self.J_J1(t, q)
        J_J2 = self.J_J2(t, q)
        J_R1 = self.J_R1(t, q)
        J_R2 = self.J_R2(t, q)
        J_C1 = J_J1 - ax2skew(r_J1C1) @ J_R1
        J_C2 = J_J2 - ax2skew(r_J2C2) @ J_R2

        # derivatives
        A_IJ1_q1 = self.A_IJ1_q1(t, q)
        n_q1 = A_IJ1_q1[:, 2, :]
        r_OJ1_q1 = self.r_OJ1_q1(t, q)
        r_OJ2_q2 = self.r_OJ2_q2(t, q)
        r_J1C1_q1 = (
            -r_OJ1_q1
            - (n @ r_J1J2) * n_q1
            - np.outer(n, r_J1J2 @ n_q1)
            + np.outer(n, n @ r_OJ1_q1)
        )
        r_J1C1_q2 = r_OJ2_q2 - np.outer(n, n @ r_OJ2_q2)
        r_J2C2_q1 = -self.radius * n_q1

        J_J1_q1 = self.J_J1_q1(t, q)
        J_J2_q2 = self.J_J2_q2(t, q)
        J_R1_q1 = self.J_R1_q1(t, q)
        J_R2_q2 = self.J_R2_q2(t, q)
        J_C1_q1 = (
            J_J1_q1
            - np.einsum("ij,jkl->ikl", ax2skew(r_J1C1), J_R1_q1)
            + np.einsum("kij,jl->ikl", ax2skew(J_R1.T), r_J1C1_q1)
        )
        J_C1_q2 = np.einsum("kij,jl->ikl", ax2skew(J_R1.T), r_J1C1_q2)
        J_C2_q1 = np.einsum("kij,jl->ikl", ax2skew(J_R2.T), r_J2C2_q1)
        J_C2_q2 = J_J2_q2 - np.einsum("ij,jkl->ikl", ax2skew(r_J2C2), J_R2_q2)

        # compute
        F = A_IJ1 @ J1_F
        F_q1 = np.einsum("ijk,j->ik", A_IJ1_q1, J1_F)

        nu1 = self._nu1
        nq1 = self._nq1
        Wla = np.zeros([self._nu], dtype=q.dtype)
        Wla[:nu1] = -F @ J_C1
        Wla[nu1:] = F @ J_C2

        Wla_q = np.zeros([self._nu, self._nq], dtype=q.dtype)
        Wla_q[:nu1, :nq1] = -np.einsum("i,ijk->jk", F, J_C1_q1) - J_C1.T @ F_q1
        Wla_q[:nu1, nq1:] = -np.einsum("i,ijk->jk", F, J_C1_q2)
        Wla_q[nu1:, :nq1] = np.einsum("i,ijk->jk", F, J_C2_q1) + J_C2.T @ F_q1
        Wla_q[nu1:, nq1:] = np.einsum("i,ijk->jk", F, J_C2_q2)

        return Wla_q

    def _KN(self, t, q, J1_F):
        A_IJ1 = self.A_IJ1(t, q)
        n = A_IJ1[:, 2]
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)
        r_J1J2 = r_OJ2 - r_OJ1
        r_J1C1 = r_J1J2 - n * (n @ r_J1J2)
        r_J2C2 = -self.radius * n

        J_J1 = self.J_J1(t, q)
        J_J2 = self.J_J2(t, q)
        J_R1 = self.J_R1(t, q)
        J_R2 = self.J_R2(t, q)
        J_J1C1 = -ax2skew(r_J1C1) @ J_R1
        J_J2C2 = -ax2skew(r_J2C2) @ J_R2
        J_C1 = J_J1 + J_J1C1
        J_C2 = J_J2 + J_J2C2

        # Delta-Jacobian of wandering points P
        # TODO: simplify outers
        Dn = -ax2skew(n) @ J_R1
        Dr_OJ1 = J_J1
        Dr_OJ2 = J_J2
        Dr_J1J2_1 = -Dr_OJ1
        Dr_J1J2_2 = Dr_OJ2
        Dr_J1C1_1 = (
            Dr_J1J2_1
            - np.outer(n, n @ Dr_J1J2_1)
            - np.outer(n, r_J1J2) @ Dn
            - (n @ r_J1J2) * Dn
        )
        Dr_J1C1_2 = Dr_J1J2_2 - np.outer(n, n @ Dr_J1J2_2)
        Dr_J2C2_1 = -self.radius * Dn

        DJ_J1 = self.J2_J1(t, q)
        DJ_J2 = self.J2_J2(t, q)
        DJ_R1 = self.J2_R1(t, q)
        DJ_R2 = self.J2_R2(t, q)
        DJ_J1C1_1 = -np.einsum("ijk,kl->jli", ax2skew(Dr_J1C1_1.T), J_R1) - np.einsum(
            "ij,jkl->ikl", ax2skew(r_J1C1), DJ_R1
        )
        DJ_J1C1_2 = -np.einsum("ijk,kl->jli", ax2skew(Dr_J1C1_2.T), J_R1)
        DJ_J2C2_1 = -np.einsum("ijk,kl->jli", ax2skew(Dr_J2C2_1.T), J_R2)
        DJ_J2C2_2 = -np.einsum("ij,jkl->ikl", ax2skew(r_J2C2), DJ_R2)
        DJ_C1_1 = DJ_J1 + DJ_J1C1_1
        DJ_C1_2 = DJ_J1C1_2
        DJ_C2_1 = DJ_J2C2_1
        DJ_C2_2 = DJ_J2 + DJ_J2C2_2

        # compute
        F = A_IJ1 @ J1_F
        D_F = -ax2skew(F) @ J_R1

        nu1 = self._nu1
        Wla = np.zeros([self._nu], dtype=q.dtype)
        Wla[:nu1] = -J_C1.T @ F
        Wla[nu1:] = J_C2.T @ F

        DWla = np.zeros((self._nu, self._nu), dtype=q.dtype)
        DWla[:nu1, :nu1] = -np.einsum("i,ijk->jk", F, DJ_C1_1) - J_C1.T @ D_F
        DWla[:nu1, nu1:] = -np.einsum("i,ijk->jk", F, DJ_C1_2)
        DWla[nu1:, :nu1] = np.einsum("i,ijk->jk", F, DJ_C2_1) + J_C2.T @ D_F
        DWla[nu1:, nu1:] = np.einsum("i,ijk->jk", F, DJ_C2_2)

        return -DWla

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
        return self._gamma(t, q, u)[2:]

    def g_N_dot_q(self, t, q, u):
        return self._gamma_q(t, q, u)[2:]

    def g_N_dot_u(self, t, q):
        return self._gamma_u(t, q)[2:]

    def W_N(self, t, q):
        return self.g_N_dot_u(t, q).T

    def g_N_ddot(self, t, q, u, u_dot):
        return self._gamma_dot(t, q, u, u_dot)[2:]

    def Wla_N_q(self, t, q, la_N):
        J1_F = np.zeros(3)
        J1_F[2:] = la_N
        return self._Wla_q(t, q, J1_F)

    def KN_N(self, t, q, la_N):
        J1_F = np.zeros(3)
        J1_F[2:] = la_N
        K = self._KN(t, q, J1_F)
        return K, np.zeros_like(K)

    ##########
    # friction
    ##########
    def W_F(self, t, q):
        return self.gamma_F_u(t, q).T

    def Wla_F_q(self, t, q, la_F):
        J1_F = np.zeros(3)
        J1_F[:2] = self.A @ la_F
        return self._Wla_q(t, q, J1_F)

    def KN_F(self, t, q, la_F):
        J1_F = np.zeros(3)
        J1_F[:2] = self.A @ la_F
        KN_test = self._KN(t, q, J1_F)
        return (KN_test + KN_test.T) / 2, (KN_test - KN_test.T) / 2

    ############
    # vtk export
    ############
    def export(self, sol_i, **kwargs):
        # extract from solution
        t = sol_i.t
        q = sol_i.q[self.qDOF]
        u = sol_i.u[self.uDOF]
        P_N = sol_i.P_N[self.la_NDOF]

        # positions and orientation
        A_IJ1 = self.A_IJ1(t, q)
        t1, t2, n = A_IJ1.T
        r_OJ1 = self.r_OJ1(t, q)
        r_OJ2 = self.r_OJ2(t, q)
        r_J1J2 = r_OJ2 - r_OJ1
        r_J1C1 = r_J1J2 - n * (n @ r_J1J2)
        r_J2C2 = -self.radius * n
        g_N = n @ r_J1J2 - self.radius

        # velocities
        v_J1 = self.v_J1(t, q, u)
        v_J2 = self.v_J2(t, q, u)
        Omega1 = self.Omega1(t, q, u)
        Omega2 = self.Omega2(t, q, u)
        v_C1 = v_J1 + cross3(Omega1, r_J1C1)
        v_C2 = v_J2 + cross3(Omega2, r_J2C2)
        _gamma = A_IJ1.T @ (v_C2 - v_C1)

        # vtk
        points = [r_OJ1 + r_J1C1, r_OJ2 + r_J2C2]
        cells = [(VTK_LINE, [0, 1])]
        point_data = dict(
            v_Ci=[v_C1, v_C2],
            Omega=[Omega1, Omega2],
            n=[n, -n],
            t1=[t1, -t1],
            t2=[t2, -t2],
            P_N=[P_N, P_N],
        )
        cell_data = dict(
            g_N=[[g_N]],
            g_N_dot=[[_gamma[2]]],
        )

        if hasattr(self, f"gamma_F"):
            P_F = sol_i.P_F[self.la_FDOF]
            cell_data["gamma_F"] = [_gamma[:2]]
            point_data["P_F"] = np.array([P_F, P_F])

        return points, cells, point_data, cell_data

    def export_blender(self, path, solution):
        nt = len(solution.t)
        r_OC1 = np.zeros((nt, 3))
        r_OC2 = np.zeros((nt, 3))
        v_C1 = np.zeros((nt, 3))
        v_C2 = np.zeros((nt, 3))
        for i in range(nt):
            t = solution.t[i]
            q = solution.q[i, self.qDOF]
            u = solution.u[i, self.uDOF]
            P_N = solution.P_N[i, self.la_NDOF]

            # positions and orientation
            A_IJ1 = self.A_IJ1(t, q)
            t1, t2, n = A_IJ1.T
            r_OJ1 = self.r_OJ1(t, q)
            r_OJ2 = self.r_OJ2(t, q)
            r_J1J2 = r_OJ2 - r_OJ1
            r_J1C1 = r_J1J2 - n * (n @ r_J1J2)
            r_J2C2 = -self.radius * n

            # velocities
            v_J1 = self.v_J1(t, q, u)
            v_J2 = self.v_J2(t, q, u)
            Omega1 = self.Omega1(t, q, u)
            Omega2 = self.Omega2(t, q, u)
            v_C1[i] = v_J1 + cross3(Omega1, r_J1C1)
            v_C2[i] = v_J2 + cross3(Omega2, r_J2C2)

            r_OC1[i] = r_OJ1 + r_J1C1
            r_OC2[i] = r_OJ2 + r_J2C2

            F2 = n * P_N
            if hasattr(self, f"gamma_F"):
                P_F = solution.P_F[i, self.la_FDOF]
                F2 += t1 * P_F[0] + t2 * P_F[1]

        make_glTF(path, f"{self.name}_C1", solution.t, r_OC1, v_C1)
        make_glTF(path, f"{self.name}_C2", solution.t, r_OC2, v_C2)
        make_glTF_arrow(path, f"{self.name}_F1", solution.t, r_OC1, r_OC1 - F2)
        make_glTF_arrow(path, f"{self.name}_F2", solution.t, r_OC2, r_OC2 + F2)
        make_glTF_arrow(path, f"{self.name}_g_N", solution.t, r_OC1, r_OC2)

    def export_blender_modes(self, path, solution):
        from warnings import warn

        warn("Sphere2Plane.export_blender_modes not implemented")
