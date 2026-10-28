import numpy as np
from cardillo.math.algebra import ax2skew

from cardillo.rods_new._cross_section import CrossSectionInertias
from cardillo.utility.coo_matrix import CooMatrix
from cardillo.utility.sparse_array_blocks import SparseArrayBlocks
from .cosseratRod_MatrixUtils import bsr_sym_skew

zeros3 = np.zeros(3, dtype=float)
eye3 = np.eye(3, dtype=float)


class CosseratRod_Inertia:
    def __init__(self, parent, mesh_kin, quadrature, projection):
        self.parent = parent

        # quadrature
        quadrature_dyn = mesh_kin.quadrature(*quadrature, 1)
        self.nquadrature_dyn_total = quadrature_dyn["nquadrature_total"]
        self.qp_dyn_vec = quadrature_dyn["qp"]
        self.qw_dyn_vec = quadrature_dyn["qw"]
        self.qels_dyn_vec = quadrature_dyn["els"]
        self.N_dyn, self.N_xi_dyn = quadrature_dyn["N"]

    def set_reference_strains(self, Q):
        _, B_gamma_bar, _ = self.parent.kinematics._eval_internal_vec(
            self.N_dyn, self.N_xi_dyn, Q
        )
        self.J_dyn_vec = np.linalg.norm(B_gamma_bar, axis=1)

        # pairs for Mass matrix
        M_pairs = [(self.N_dyn, self.N_dyn, self.qw_dyn_vec * self.J_dyn_vec)]
        nu_node = self.parent.velocity.nu_node
        self.M_SAB = SparseArrayBlocks(
            (self.parent.nu, self.parent.nu), (nu_node, nu_node), M_pairs
        )

    def set_cross_section_inertias(self, cross_section_inertias):
        if cross_section_inertias == False:
            self.include_f_gyr = False
            self.cross_section_inertias = CrossSectionInertias()
        else:
            self.include_f_gyr = True
            self.cross_section_inertias = cross_section_inertias

        self.A_rho0_qp = self.cross_section_inertias.A_rho0(self.qp_dyn_vec)
        self.B_I_rho0_qp = self.cross_section_inertias.B_I_rho0(self.qp_dyn_vec)

    def _assert_nodal_quadrature(self, quadrature, name):
        assert (
            quadrature[1] == "Trapezoidal"
        ), f"{name} dynamics are only implemented for the trapezoidal rule!"
        # all quadrature points have to coincide with nodes
        N_dyn = self.N_dyn.toarray()
        assert np.allclose(N_dyn.max(axis=1), 1.0) and np.allclose(
            np.abs(N_dyn).sum(axis=1), 1.0
        ), f"{name} dynamics require all quadrature points to coincide with the nodes!"


class CosseratRod_dynamics_PG_IB(CosseratRod_Inertia):
    def __init__(self, parent, mesh_kin, quadrature, projection):
        super().__init__(parent, mesh_kin, quadrature, projection)
        self.parent.constant_mass_matrix = True
        self.parent.linear_momentum = self.linear_momentum
        self.parent.angular_momentum = self.angular_momentum
        self.parent.E_kin = self.E_kin
        self.parent.M = self.M

        self.nnodes = self.parent.nnodes

    def linear_momentum(self, t, q, u):
        unodes = u.reshape(self.nnodes, -1)
        v = self.N_dyn @ unodes[:, :3]
        linear_momentum = np.sum(
            v * (self.A_rho0_qp * self.J_dyn_vec * self.qw_dyn_vec)[:, None], axis=0
        )
        return linear_momentum

    def angular_momentum(self, t, q, u):
        r_OC, A_IB = self.parent.kinematics._eval_vec(self.N_dyn, q)

        unodes = u.reshape(self.nnodes, -1)
        vO = self.N_dyn @ unodes
        v_C = vO[:, :3]
        B_Omega = vO[:, 3:]

        angular_momentum_qp = np.cross(r_OC, v_C) * self.A_rho0_qp[:, None] + np.einsum(
            "ijk,ikl,il->ij", A_IB, self.B_I_rho0_qp, B_Omega
        )
        angular_momentum = np.sum(
            angular_momentum_qp * (self.J_dyn_vec * self.qw_dyn_vec)[:, None], axis=0
        )
        return angular_momentum

    def E_kin(self, t, q, u):
        unodes = u.reshape(self.nnodes, -1)
        vO = self.N_dyn @ unodes

        v = vO[:, :3]
        B_Omega = vO[:, 3:]

        E_kin_i = 0.5 * (
            self.A_rho0_qp * np.sum(v * v, axis=1)
            + np.einsum("ij,ijk,ik->i", B_Omega, self.B_I_rho0_qp, B_Omega)
        )

        E_kin = np.sum(E_kin_i * self.qw_dyn_vec * self.J_dyn_vec)
        return E_kin

    def M(self, t, q):
        return self.__M

    def assembler_callback(self):
        # TODO: make this sparse?
        M_qp = np.empty((1, self.nquadrature_dyn_total, 6, 6))
        M_qp[0, :, :3, :3] = eye3 * self.A_rho0_qp[:, None, None]
        M_qp[0, :, :3, 3:] = 0.0
        M_qp[0, :, 3:, :3] = 0.0
        M_qp[0, :, 3:, 3:] = self.B_I_rho0_qp

        self.__M = self.M_SAB.add_blocks(M_qp)
        return self.__M

    def f_gyr(self, t, q, u):
        unodes = u.reshape(self.nnodes, -1)
        B_Omega = self.N_dyn @ unodes[:, 3:]

        # spin
        B_L_qp = np.einsum("ijk,ik->ij", self.B_I_rho0_qp, B_Omega)
        f_gyr_qp = np.cross(B_Omega, B_L_qp)

        f_gyr = np.empty((self.nnodes, 6))
        f_gyr[:, :3] = 0.0
        f_gyr[:, 3:] = self.N_dyn.T @ (
            f_gyr_qp * (-self.J_dyn_vec * self.qw_dyn_vec)[:, None]
        )
        return f_gyr.reshape(-1)

    def f_gyr_u(self, t, q, u):
        unodes = u.reshape(self.nnodes, -1)
        B_Omega = self.N_dyn @ unodes[:, 3:]

        # spin
        B_L_qp = np.einsum("ijk,ik->ij", self.B_I_rho0_qp, B_Omega)

        f_gyr_qp_ubar = np.zeros((self.nquadrature_dyn_total, 6, 6))
        # B_Omega_tilde @ B_I_rho0 - B_L_tilde
        f_gyr_qp_ubar[:, 3:, 3:] = np.cross(
            self.B_I_rho0_qp, B_Omega[:, :, None], axisa=1, axisb=1, axisc=1
        ) + ax2skew(B_L_qp)

        return self.M_SAB.add_blocks(np.array([f_gyr_qp_ubar]))

    def DG_f_gyr(self, t, q, u):
        # D + G = -f_gyr_u, split into symmetric and skew-symmetric part
        return bsr_sym_skew(-self.f_gyr_u(t, q, u))


class CosseratRod_dynamics_BG(CosseratRod_Inertia):
    """Dynamics of the Bubnov-Galerkin formulation for the R12 parametrization,
    where u = q_dot = (r_dot, d1_dot, d2_dot, d3_dot) is interpolated as q. With
    the director inertia B_E_rho0 = 0.5 * trace(B_I_rho0) * eye3 - B_I_rho0, the
    kinetic energy of the cross-section is
    0.5 * (A_rho0 * r_dot @ r_dot + B_E_rho0_ij * di_dot @ dj_dot),
    which coincides with 0.5 * (A_rho0 * r_dot @ r_dot + B_Omega @ B_I_rho0 @ B_Omega)
    for orthonormal directors. Hence, the mass matrix is constant and there are
    no gyroscopic forces for any quadrature rule."""

    def __init__(self, parent, mesh_kin, quadrature, projection):
        super().__init__(parent, mesh_kin, quadrature, projection)

        # constant mass matrix only for R12, quaternions would require a
        # configuration dependent mass matrix and gyroscopic forces
        if self.parent.kinematics.nq_node == 12:
            self.parent.constant_mass_matrix = True

        self.statics_only = self.parent.kinematics.nq_node != 12
        if self.statics_only:
            return

        self.parent.linear_momentum = self.linear_momentum
        self.parent.angular_momentum = self.angular_momentum
        self.parent.E_kin = self.E_kin
        self.parent.M = self.M

        self.nnodes = self.parent.nnodes

    def set_reference_strains(self, Q):
        if self.statics_only:
            return
        super().set_reference_strains(Q)

    def set_cross_section_inertias(self, cross_section_inertias):
        if self.statics_only:
            if cross_section_inertias is not False:
                raise NotImplementedError(
                    "BG with Quaternion is only implemented for statics, "
                    "i.e., cross_section_inertias=False!"
                )
            self.include_f_gyr = False
            return

        super().set_cross_section_inertias(cross_section_inertias)
        # there are no gyroscopic forces
        self.include_f_gyr = False

        # director inertia
        trace = np.trace(self.B_I_rho0_qp, axis1=1, axis2=2)
        self.B_E_rho0_qp = 0.5 * trace[:, None, None] * eye3 - self.B_I_rho0_qp

    def _eval_qp(self, q, u):
        r_OC, d = np.split(self.N_dyn @ q.reshape(self.nnodes, -1), [3], axis=1)
        v_C, d_dot = np.split(self.N_dyn @ u.reshape(self.nnodes, -1), [3], axis=1)
        return r_OC, d.reshape(-1, 3, 3), v_C, d_dot.reshape(-1, 3, 3)

    def linear_momentum(self, t, q, u):
        _, _, v_C, _ = self._eval_qp(q, u)
        return np.sum(
            v_C * (self.A_rho0_qp * self.J_dyn_vec * self.qw_dyn_vec)[:, None], axis=0
        )

    def angular_momentum(self, t, q, u):
        r_OC, d, v_C, d_dot = self._eval_qp(q, u)
        # B_E_rho0_ij * di x dj_dot
        angular_momentum_qp = np.cross(r_OC, v_C) * self.A_rho0_qp[:, None] + np.einsum(
            "pij,pijk->pk",
            self.B_E_rho0_qp,
            np.cross(d[:, :, None, :], d_dot[:, None, :, :]),
        )
        return np.sum(
            angular_momentum_qp * (self.J_dyn_vec * self.qw_dyn_vec)[:, None], axis=0
        )

    def E_kin(self, t, q, u):
        _, _, v_C, d_dot = self._eval_qp(q, u)
        E_kin_i = 0.5 * (
            self.A_rho0_qp * np.sum(v_C * v_C, axis=1)
            + np.einsum("pij,pik,pjk->p", self.B_E_rho0_qp, d_dot, d_dot)
        )
        return np.sum(E_kin_i * self.qw_dyn_vec * self.J_dyn_vec)

    def M(self, t, q):
        return self.__M

    def assembler_callback(self):
        if self.statics_only:
            return

        nqp = self.nquadrature_dyn_total
        M_qp = np.zeros((1, nqp, 12, 12))
        M_qp[0, :, :3, :3] = eye3 * self.A_rho0_qp[:, None, None]
        M_qp[0, :, 3:, 3:] = np.einsum("pij,kl->pikjl", self.B_E_rho0_qp, eye3).reshape(
            nqp, 9, 9
        )

        self.__M = self.M_SAB.add_blocks(M_qp)
        return self.__M


class CosseratRod_dynamics_DBG(CosseratRod_dynamics_PG_IB):
    """Dynamics of the Bubnov-Galerkin formulation with discrete nullspace
    projection. With the trapezoidal rule, the quadrature points coincide with
    the nodes, where the velocities are given by u. Hence, it is identical to
    the Petrov-Galerkin formulation."""

    def __init__(self, parent, mesh_kin, quadrature, projection):
        super().__init__(parent, mesh_kin, quadrature, projection)
        self._assert_nodal_quadrature(quadrature, "DBG")
