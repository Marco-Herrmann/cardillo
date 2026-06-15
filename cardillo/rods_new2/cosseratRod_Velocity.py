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
from cardillo.utility.coo_matrix import CooMatrix
from cardillo.utility.sparse_array_blocks import SparseArrayBlocks

zeros3 = np.zeros(3, dtype=float)
eye3 = np.eye(3, dtype=float)


class Rod_Velocity(ABC):
    def J_P(self, t, qi, xi, B_r_CP=zeros3):
        return self.velocity.J_P(t, qi, xi, B_r_CP)

    def J_P_q(self, t, qi, xi, B_r_CP=zeros3):
        return self.velocity.J_P_q(t, qi, xi, B_r_CP)

    def J2_P(self, t, qi, xi, B_r_CP=zeros3):
        return self.velocity.J2_P(t, qi, xi, B_r_CP)

    def v_P(self, t, qi, ui, xi, B_r_CP=zeros3):
        return self.velocity.v_P(t, qi, ui, xi, B_r_CP)

    def v_P_q(self, t, qi, ui, xi, B_r_CP=zeros3):
        return self.velocity.v_P_q(t, qi, ui, xi, B_r_CP)

    def a_P(self, t, qi, ui, ui_dot, xi, B_r_CP=zeros3):
        return self.velocity.a_P(t, qi, ui, ui_dot, xi, B_r_CP)

    def a_P_q(self, t, qi, ui, ui_dot, xi, B_r_CP=zeros3):
        return self.velocity.a_P_q(t, qi, ui, ui_dot, xi, B_r_CP)

    def a_P_u(self, t, qi, ui, ui_dot, xi, B_r_CP=zeros3):
        return self.velocity.a_P_u(t, qi, ui, ui_dot, xi, B_r_CP)

    def B_J_R(self, t, qi, xi):
        return self.velocity.B_J_R(t, qi, xi)

    def B_J_R_q(self, t, qi, xi):
        return self.velocity.B_J_R_q(t, qi, xi)

    def B_J2_R(self, t, qi, xi):
        return self.velocity.B_J2_R(t, qi, xi)

    def B_Omega(self, t, qi, ui, xi):
        return self.velocity.B_Omega(t, qi, ui, xi)

    def B_Omega_q(self, t, qi, ui, xi):
        return self.velocity.B_Omega_q(t, qi, ui, xi)

    def B_Psi(self, t, qi, ui, ui_dot, xi):
        return self.velocity.B_Psi(t, qi, ui, ui_dot, xi)

    def B_Psi_q(self, t, qi, ui, ui_dot, xi):
        return self.velocity.B_Psi_q(t, qi, ui, ui_dot, xi)

    def B_Psi_u(self, t, qi, ui, ui_dot, xi):
        return self.velocity.B_Psi_u(t, qi, ui, ui_dot, xi)


class CosseratRod_Velocity(ABC):
    def __init__(self, parent, mesh):
        self.parent = parent
        self.mesh = mesh


class CosseratRod_PG_IB(CosseratRod_Velocity):
    def __init__(self, parent, mesh):
        super().__init__(parent, mesh)
        self.nu_node = 6

    def J_P(self, t, qi, xi, B_r_CP=zeros3):
        point_dict = self.parent.get_interaction_point(xi)
        J_C = point_dict["Nu"][:3]
        if B_r_CP @ B_r_CP == 0.0:
            return J_C

        B_J_R = point_dict["Nu"][3:]
        B_J_CP = np.cross(-B_r_CP[:, None], B_J_R, axisa=0, axisb=0, axisc=0)
        A_IB = self.parent.A_IB(t, qi, xi)
        return J_C + A_IB @ B_J_CP

    def J_P_q(self, t, qi, xi, B_r_CP=zeros3):
        point_dict = self.parent.get_interaction_point(xi)
        if B_r_CP @ B_r_CP == 0.0:
            return point_dict["zero_3_nui_nqi"]

        B_J_R = point_dict["Nu"][3:]
        B_J_CP = np.cross(-B_r_CP[:, None], B_J_R, axisa=0, axisb=0, axisc=0)
        A_IB_q = self.parent.A_IB_q(t, qi, xi)
        return np.einsum("ijk, jl -> ilk", A_IB_q, B_J_CP)

    def v_P(self, t, qi, ui, xi, B_r_CP=zeros3):
        point_dict = self.parent.get_interaction_point(xi)
        N = point_dict["N"]
        unodes = ui.reshape(point_dict["nnodes"], -1)
        if B_r_CP @ B_r_CP == 0.0:
            return N @ unodes[:, :3]

        vO = N @ unodes
        B_v_CP = np.cross(vO[3:], B_r_CP)
        A_IB = self.parent.A_IB(t, qi, xi)
        return vO[:3] + A_IB @ B_v_CP

    def v_P_q(self, t, qi, ui, xi, B_r_CP=zeros3):
        point_dict = self.parent.get_interaction_point(xi)
        if B_r_CP @ B_r_CP == 0.0:
            return point_dict["zero_3_nqi"]

        N = point_dict["N"]
        unodes = ui.reshape(point_dict["nnodes"], -1)
        B_Omega = N @ unodes[:, 3:]
        B_v_CP = np.cross(B_Omega, B_r_CP)
        A_IB_q = self.parent.A_IB_q(t, qi, xi)
        return np.einsum("ijk,j->ik", A_IB_q, B_v_CP)

    def a_P(self, t, qi, ui, ui_dot, xi, B_r_CP=zeros3):
        point_dict = self.parent.get_interaction_point(xi)
        N = point_dict["N"]
        u_dotnodes = ui_dot.reshape(point_dict["nnodes"], -1)
        if B_r_CP @ B_r_CP == 0.0:
            return N @ u_dotnodes[:, :3]

        unodes = ui.reshape(point_dict["nnodes"], -1)
        B_Omega = N @ unodes[:, 3:]
        aP = N @ u_dotnodes
        B_a_CP = np.cross(aP[3:], B_r_CP) + np.cross(B_Omega, np.cross(B_Omega, B_r_CP))
        A_IB = self.parent.A_IB(t, qi, xi)
        return aP[:3] + A_IB @ B_a_CP

    def a_P_q(self, t, qi, ui, ui_dot, xi, B_r_CP=zeros3):
        point_dict = self.parent.get_interaction_point(xi)
        if B_r_CP @ B_r_CP == 0.0:
            return point_dict["zero_3_nqi"]

        N = point_dict["N"]
        u_dotnodes = ui_dot.reshape(point_dict["nnodes"], -1)
        unodes = ui.reshape(point_dict["nnodes"], -1)
        B_Omega = N @ unodes[:, 3:]
        aP = N @ u_dotnodes
        B_a_CP = np.cross(aP[3:], B_r_CP) + np.cross(B_Omega, np.cross(B_Omega, B_r_CP))
        A_IB_q = self.parent.A_IB_q(t, qi, xi)
        return np.einsum("ijk,j->ik", A_IB_q, B_a_CP)

    def a_P_u(self, t, qi, ui, ui_dot, xi, B_r_CP=zeros3):
        point_dict = self.parent.get_interaction_point(xi)
        if B_r_CP @ B_r_CP == 0.0:
            return point_dict["zero_3_nui"]

        N = point_dict["N"]
        unodes = ui.reshape(point_dict["nnodes"], -1)
        B_Omega = N @ unodes[:, 3:]
        B_a_CP_B_Omega = ax2skew(np.cross(B_r_CP, B_Omega)) - ax2skew(
            B_Omega
        ) @ ax2skew(B_r_CP)
        A_IB = self.parent.A_IB(t, qi, xi)
        return A_IB @ B_a_CP_B_Omega @ point_dict["Nu"][3:]

    def B_J_R(self, t, qi, xi):
        point_dict = self.parent.get_interaction_point(xi)
        return point_dict["Nu"][3:]

    def B_J_R_q(self, t, qi, xi):
        point_dict = self.parent.get_interaction_point(xi)
        return point_dict["zero_3_nui_nqi"]

    def B_Omega(self, t, qi, ui, xi):
        point_dict = self.parent.get_interaction_point(xi)
        N = point_dict["N"]
        unodes = ui.reshape(point_dict["nnodes"], -1)
        return N @ unodes[:, 3:]

    def B_Omega_q(self, t, qi, ui, xi):
        point_dict = self.parent.get_interaction_point(xi)
        return point_dict["zero_3_nqi"]

    def B_Psi(self, t, qi, ui, ui_dot, xi):
        point_dict = self.parent.get_interaction_point(xi)
        N = point_dict["N"]
        u_dotnodes = ui_dot.reshape(point_dict["nnodes"], -1)
        return N @ u_dotnodes[:, 3:]

    def B_Psi_q(self, t, qi, ui, ui_dot, xi):
        point_dict = self.parent.get_interaction_point(xi)
        return point_dict["zero_3_nqi"]

    def B_Psi_u(self, t, qi, ui, ui_dot, xi):
        point_dict = self.parent.get_interaction_point(xi)
        return point_dict["zero_3_nui"]
