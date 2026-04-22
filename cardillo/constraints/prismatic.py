import numpy as np
from cardillo.constraints._base import ProjectedPositionOrientationBase
from cardillo.math import cross3, ax2skew
from cardillo.math.approx_fprime import approx_fprime


class Prismatic(ProjectedPositionOrientationBase):
    def __init__(
        self,
        subsystem1,
        subsystem2,
        axis,
        r_OJ0=None,
        A_IJ0=None,
        xi1=None,
        xi2=None,
        name="prismatic",
    ):
        assert axis in (0, 1, 2)
        self.axis = axis

        # remove free axis
        constrained_axes_displacement = np.delete((0, 1, 2), axis)

        # all orientations are constrained
        projection_pairs_rotation = [(0, 1), (1, 2), (2, 0)]

        super().__init__(
            subsystem1,
            subsystem2,
            r_OJ0=r_OJ0,
            A_IJ0=A_IJ0,
            constrained_axes_translation=constrained_axes_displacement,
            projection_pairs_rotation=projection_pairs_rotation,
            xi1=xi1,
            xi2=xi2,
            name=name,
        )

    def l(self, t, q):
        A_IJ1 = self.A_IJ1(t, q)
        r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
        return r_J1J2 @ A_IJ1[:, self.axis]

    def l_q(self, t, q):
        # return approx_fprime(q, lambda q: self.l(t, q), method="3-point", eps=1e-6)

        nq1 = self._nq1
        g_q = np.zeros((1, self._nq), dtype=q.dtype)

        A_IJ1 = self.A_IJ1(t, q)
        A_IJ1_q1 = self.A_IJ1_q1(t, q)

        r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
        r_OJ1_q1 = self.r_OJ1_q1(t, q)
        r_OJ2_q2 = self.r_OJ2_q2(t, q)
        g_q[0, :nq1] = -A_IJ1[:, self.axis] @ r_OJ1_q1 + r_J1J2 @ A_IJ1_q1[:, self.axis]
        g_q[0, nq1:] = A_IJ1[:, self.axis] @ r_OJ2_q2

        return g_q

        # g_q_num = approx_fprime(
        #     q, lambda q: self.g(t, q), method="cs", eps=1e-12
        # )
        # diff = g_q - g_q_num
        # error = np.linalg.norm(diff)
        # print(f"error g_q: {error}")
        # return g_q_num

    def l_dot(self, t, q, u):
        raise NotImplementedError
        g_dot = np.zeros(self.nla_g, dtype=np.common_type(q, u))

        A_IJ1 = self.A_IJ1(t, q)
        Omega1 = self.Omega1(t, q, u)
        if self.constrain_translation:
            r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
            v_J1J2 = self.v_J2(t, q, u) - self.v_J1(t, q, u)
            for i, ax in enumerate(self.constrained_axes_displacement):
                g_dot[i] = A_IJ1[:, ax] @ v_J1J2 + cross3(A_IJ1[:, ax], r_J1J2) @ Omega1

        if self.constrain_orientation:
            A_IJ2 = self.A_IJ2(t, q)
            Omega21 = Omega1 - self.Omega2(t, q, u)

            for i, (a, b) in enumerate(self.projection_pairs_rotation):
                n = cross3(A_IJ1[:, a], A_IJ2[:, b])
                g_dot[self.nla_g_trans + i] = n @ Omega21

        return g_dot

    def l_dot_q(self, t, q, u):
        raise NotImplementedError
        nq1 = self._nq1
        g_dot_q = np.zeros((self.nla_g, self._nq), dtype=np.common_type(q, u))

        A_IJ1 = self.A_IJ1(t, q)
        A_IJ1_q1 = self.A_IJ1_q1(t, q)
        Omega1 = self.Omega1(t, q, u)
        Omega1_q1 = self.Omega1_q1(t, q, u)
        if self.constrain_translation:
            r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
            v_J1J2 = self.v_J2(t, q, u) - self.v_J1(t, q, u)
            r_OJ1_q1 = self.r_OJ1_q1(t, q)
            r_OJ2_q2 = self.r_OJ2_q2(t, q)
            v_J1_q1 = self.v_J1_q1(t, q, u)
            v_J2_q2 = self.v_J2_q2(t, q, u)
            for i, ax in enumerate(self.constrained_axes_displacement):
                g_dot_q[i, :nq1] = (
                    -A_IJ1[:, ax] @ v_J1_q1
                    + cross3(A_IJ1[:, ax], r_J1J2) @ Omega1_q1
                    + (v_J1J2 + cross3(r_J1J2, Omega1)) @ A_IJ1_q1[:, ax]
                    - cross3(Omega1, A_IJ1[:, ax]) @ r_OJ1_q1
                )
                g_dot_q[i, nq1:] = (
                    A_IJ1[:, ax] @ v_J2_q2 + cross3(Omega1, A_IJ1[:, ax]) @ r_OJ2_q2
                )

        if self.constrain_orientation:
            A_IJ2 = self.A_IJ2(t, q)
            A_IJ2_q2 = self.A_IJ2_q2(t, q)

            Omega21 = Omega1 - self.Omega2(t, q, u)
            Omega2_q2 = self.Omega2_q2(t, q, u)

            for i, (a, b) in enumerate(self.projection_pairs_rotation):
                e_a, e_b = A_IJ1[:, a], A_IJ2[:, b]
                n = cross3(e_a, e_b)
                g_dot_q[self.nla_g_trans + i, :nq1] = (
                    n @ Omega1_q1 - Omega21 @ ax2skew(e_b) @ A_IJ1_q1[:, a]
                )
                g_dot_q[self.nla_g_trans + i, nq1:] = (
                    -n @ Omega2_q2 + Omega21 @ ax2skew(e_a) @ A_IJ2_q2[:, b]
                )

        return g_dot_q

        # g_dot_q_num = approx_fprime(
        #     q, lambda q: self.g_dot(t, q, u), method="cs", eps=1e-12
        # )
        # diff = g_dot_q - g_dot_q_num
        # error = np.linalg.norm(diff)
        # print(f"error g_dot_q: {error}")
        # return g_dot_q_num

    def l_dot_u(self, t, q):
        return self.W_l(t, q).T

    def l_ddot(self, t, q, u, u_dot):
        raise NotImplementedError
        g_ddot = np.zeros(self.nla_g, dtype=np.common_type(q, u, u_dot))

        A_IJ1 = self.A_IJ1(t, q)
        Omega1 = self.Omega1(t, q, u)
        Psi1 = self.Psi1(t, q, u, u_dot)
        if self.constrain_translation:
            r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
            v_J1J2 = self.v_J2(t, q, u) - self.v_J1(t, q, u)
            a_J1J2 = self.a_J2(t, q, u, u_dot) - self.a_J1(t, q, u, u_dot)
            for i, ax in enumerate(self.constrained_axes_displacement):
                e_dot = cross3(Omega1, A_IJ1[:, ax])
                g_ddot[i] = (
                    A_IJ1[:, ax] @ a_J1J2
                    + v_J1J2 @ e_dot
                    + cross3(A_IJ1[:, ax], r_J1J2) @ Psi1
                    + cross3(A_IJ1[:, ax], v_J1J2) @ Omega1
                    + cross3(e_dot, r_J1J2) @ Omega1
                )

        if self.constrain_orientation:
            A_IJ2 = self.A_IJ2(t, q)
            Omega2 = self.Omega2(t, q, u)
            Omega21 = Omega1 - Omega2
            Psi21 = Psi1 - self.Psi2(t, q, u, u_dot)

            for i, (a, b) in enumerate(self.projection_pairs_rotation):
                e_a, e_b = A_IJ1[:, a], A_IJ2[:, b]
                n = cross3(e_a, e_b)
                g_ddot[self.nla_g_trans + i] = (
                    cross3(cross3(Omega1, e_a), e_b) + cross3(e_a, cross3(Omega2, e_b))
                ) @ Omega21 + n @ Psi21

        return g_ddot

    def W_l(self, t, q):
        nu1 = self._nu1
        W_l = np.zeros((self._nu), dtype=q.dtype)

        A_IJ1 = self.A_IJ1(t, q)
        J_R1 = self.J_R1(t, q)
        r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
        J_J1 = self.J_J1(t, q)
        J_J2 = self.J_J2(t, q)
        W_l[:nu1] = (
            -A_IJ1[:, self.axis] @ J_J1 + cross3(A_IJ1[:, self.axis], r_J1J2) @ J_R1
        )
        W_l[nu1:] = A_IJ1[:, self.axis] @ J_J2
        
        return W_l

    def W_l_q(self, t, q):
        raise NotImplementedError
        W_l_q_num = approx_fprime(
            # q, lambda q: self.W_g(t, q), method="3-point", eps=1e-6
            q, lambda q: self.W_l(t, q), method="cs", eps=1e-12
        )
        return W_l_q_num
    
        nq1 = self._nq1
        nu1 = self._nu1
        W_l_q = np.zeros((self._nu, 1, self._nq), dtype=q.dtype)

        A_IJ1 = self.A_IJ1(t, q)
        A_IJ1_q1 = self.A_IJ1_q1(t, q)
        J_R1 = self.J_R1(t, q)
        J_R1_q1 = self.J_R1_q1(t, q)
        
        r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
        r_OJ1_q1 = self.r_OJ1_q1(t, q)
        r_OJ2_q2 = self.r_OJ2_q2(t, q)
        J_J1 = self.J_J1(t, q)
        J_J2 = self.J_J2(t, q)
        J_J1_q1 = self.J_J1_q1(t, q)
        J_J2_q2 = self.J_J2_q2(t, q)
        for i, ax in enumerate(self.constrained_axes_displacement):
            W_l_q[:nu1, :nq1] += (
                np.einsum("i,ijk->jk", -la_g[i] * A_IJ1[:, ax], J_J1_q1)
                + np.einsum("ik,ij->jk", -la_g[i] * A_IJ1_q1[:, ax], J_J1)
                + np.einsum(
                    "i,ijk->jk", la_g[i] * cross3(A_IJ1[:, ax], r_J1J2), J_R1_q1
                )
                + np.einsum(
                    "ik,ij->jk", -la_g[i] * ax2skew(A_IJ1[:, ax]) @ r_OJ1_q1, J_R1
                )
                + np.einsum(
                    "ik,ij->jk", -la_g[i] * ax2skew(r_J1J2) @ A_IJ1_q1[:, ax], J_R1
                )
            )
            Wla_g_q[:nu1, nq1:] += np.einsum(
                "ik,ij->jk", la_g[i] * ax2skew(A_IJ1[:, ax]) @ r_OJ2_q2, J_R1
            )

            Wla_g_q[nu1:, :nq1] += np.einsum(
                "ij,ik->kj", la_g[i] * A_IJ1_q1[:, ax], J_J2
            )
            Wla_g_q[nu1:, nq1:] += np.einsum(
                "i,ijk->jk", la_g[i] * A_IJ1[:, ax], J_J2_q2
            )
        return Wla_g_q

        # Wla_g_q_num = approx_fprime(
        #     # q, lambda q: self.W_g(t, q) @ la_g, method="3-point", eps=1e-6
        #     q, lambda q: self.W_g(t, q) @ la_g, method="cs", eps=1e-12
        # )
        # diff = Wla_g_q - Wla_g_q_num
        # error = np.linalg.norm(diff)
        # # if error > 1.0e-8:
        # print(f"error Wla_g_q: {error}")

        # return Wla_g_q_num

    def Wla_l_q(self, t, q, la_l): 
        nq1 = self._nq1
        nu1 = self._nu1
        Wla_l_q = np.zeros((self._nu, self._nq), dtype=q.dtype)

        A_IJ1 = self.A_IJ1(t, q)
        A_IJ1_q1 = self.A_IJ1_q1(t, q)
        J_R1 = self.J_R1(t, q)
        J_R1_q1 = self.J_R1_q1(t, q)
        
        r_J1J2 = self.r_OJ2(t, q) - self.r_OJ1(t, q)
        r_OJ1_q1 = self.r_OJ1_q1(t, q)
        r_OJ2_q2 = self.r_OJ2_q2(t, q)
        J_J1 = self.J_J1(t, q)
        J_J2 = self.J_J2(t, q)
        J_J1_q1 = self.J_J1_q1(t, q)
        J_J2_q2 = self.J_J2_q2(t, q)
        
        Wla_l_q[:nu1, :nq1] += (
            np.einsum("i,ijk->jk", -la_l * A_IJ1[:, self.axis], J_J1_q1)
            + np.einsum("ik,ij->jk", -la_l * A_IJ1_q1[:, self.axis], J_J1)
            + np.einsum(
                "i,ijk->jk", la_l * cross3(A_IJ1[:, self.axis], r_J1J2), J_R1_q1
            )
            + np.einsum(
                "ik,ij->jk", -la_l * ax2skew(A_IJ1[:, self.axis]) @ r_OJ1_q1, J_R1
            )
            + np.einsum(
                "ik,ij->jk", -la_l * ax2skew(r_J1J2) @ A_IJ1_q1[:, self.axis], J_R1
            )
        )
        Wla_l_q[:nu1, nq1:] += np.einsum(
            "ik,ij->jk", la_l * ax2skew(A_IJ1[:, self.axis]) @ r_OJ2_q2, J_R1
        )

        Wla_l_q[nu1:, :nq1] += np.einsum(
            "ij,ik->kj", la_l * A_IJ1_q1[:, self.axis], J_J2
        )
        Wla_l_q[nu1:, nq1:] += np.einsum(
            "i,ijk->jk", la_l * A_IJ1[:, self.axis], J_J2_q2
        )
        return Wla_l_q



