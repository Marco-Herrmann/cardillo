from abc import ABC, abstractmethod
import numpy as np
from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey
from scipy.sparse import (
    block_diag,
    bsr_array,
    csr_array,
    eye_array,
)
from scipy.sparse.linalg import spsolve
from warnings import warn

from cardillo.math.algebra import norm, cross3, ax2skew, ax2skew_a
from cardillo.math.approx_fprime import approx_fprime
from cardillo.math.rotations import (
    Log_SO3_quat,
    Exp_SO3_quat,
    Exp_SO3_quat_P,
    T_SO3_quat,
    T_SO3_quat_P,
    T_SO3_inv_quat,
    T_SO3_inv_quat_P,
    Log_SO3_R9,
    Exp_SO3_R9,
    Exp_SO3_R9_R9,
    T_SO3_R9,
    T_SO3_R9_R9,
    T_SO3_inv_R9,
    T_SO3_inv_R9_R9,
)
from cardillo.rods_new._cross_section import CrossSectionInertias
from cardillo.utility.coo_matrix import CooMatrix
from cardillo.utility.sparse_array_blocks import SparseArrayBlocks

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
        self.M_h_u_SAB = SparseArrayBlocks(
            (self.parent.nu, self.parent.nu), (6, 6), M_pairs
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

        self.__M = self.M_h_u_SAB.add_blocks(M_qp)
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

        return self.M_h_u_SAB.add_blocks(np.array([f_gyr_qp_ubar]))

    def DG_f_gyr(self, t, q, u):
        if u @ u > 0.0:
            print("DG_f_gyr not implemented for non-zero u!")
        D = CooMatrix((self.parent.nu, self.parent.nu))
        return D, D


class CosseratRod_dynamics_BG(CosseratRod_Inertia): ...
