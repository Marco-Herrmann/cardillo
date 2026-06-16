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


class CosseratRod_Velocity(ABC):
    def __init__(self, parent, mesh):
        self.parent = parent
        self.mesh = mesh

        # add methods to parent
        self.parent.J_P = self.J_P
        self.parent.J_P_q = self.J_P_q
        self.parent.J2_P = self.J2_P
        self.parent.v_P = self.v_P
        self.parent.v_P_q = self.v_P_q
        self.parent.a_P = self.a_P
        self.parent.a_P_q = self.a_P_q
        self.parent.a_P_u = self.a_P_u

        self.parent.B_J_R = self.B_J_R
        self.parent.B_J_R_q = self.B_J_R_q
        self.parent.B_J2_R = self.B_J2_R
        self.parent.B_Omega = self.B_Omega
        self.parent.B_Omega_q = self.B_Omega_q
        self.parent.B_Psi = self.B_Psi
        self.parent.B_Psi_q = self.B_Psi_q
        self.parent.B_Psi_u = self.B_Psi_u


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

    def J2_P(self, t, qi, xi, B_r_CP=zeros3):
        point_dict = self.parent.get_interaction_point(xi)
        if B_r_CP @ B_r_CP == 0.0:
            return point_dict["zero_3_nui_nui"]

        qnodes = qi.reshape(point_dict["nnodes"], -1)
        N = point_dict["N"]

        A_IB = self._A_IB(N @ qnodes[:, 3:])
        B_J2_R_phi = -0.5 * ax2skew_a()
        B_r_CP_tilde = ax2skew(B_r_CP)

        # TODO: implement for xi at an really arbitrary point
        nnodes = point_dict["nnodes"]
        assert nnodes == 1

        # only operations on relevant DOFs and using B_J_R = [zero, eye]
        J2_P = np.zeros((3, 6, 6), dtype=qi.dtype)
        J2_P[:, 3:, 3:] = np.einsum(
            "jl, lki -> ijk", B_r_CP_tilde, ax2skew_a() @ A_IB.T
        ) - np.einsum("il, ljk -> ijk", A_IB @ B_r_CP_tilde, B_J2_R_phi)
        return J2_P

        # TODO: implement for B_r_CP != 0.0
        assert np.linalg.norm(B_r_CP) == 0.0
        point_dict = self.get_interaction_point(xi)
        return point_dict["zero_3_nui_nui"]

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

    def B_J2_R(self, t, qi, xi):
        point_dict = self.parent.get_interaction_point(xi)
        # N = point_dict["N"]
        # qnodes = qi.reshape(point_dict["nnodes"], -1)

        assert point_dict["nnodes"] == 1

        B_J2_R = np.zeros((3, 6, 6), dtype=qi.dtype)
        B_J2_R[:, 3:, 3:] = -0.5 * ax2skew_a()
        return B_J2_R

        z = point_dict["zero_3_nui_nui"]
        warn("B_J2_R not implemented yet")
        return z
        return point_dict["B_J2_R"]

        # TODO: check plus-minus bzw. [:, None]
        N_p, _ = self.basis_functions_p(xi)
        B_J2_R = np.zeros((3, self.nu_element, self.nu_element), dtype=q.dtype)
        for node in range(self.nnodes_element_p):
            DOF = self.nodalDOF_element_p_u[node]
            B_J2_R[:, DOF, DOF[:, None]] = -0.5 * N_p[node] * ax2skew_a()
        return B_J2_R

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
