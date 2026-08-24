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
from cardillo.utility.check_time_derivatives import check_time_derivatives
from cardillo.utility.coo_matrix import CooMatrix
from cardillo.utility.sparse_array_blocks import SparseArrayBlocks

zeros3 = np.zeros(3, dtype=float)
eye3 = np.eye(3, dtype=float)


class Rod_Kinematics(ABC):
    def r_OP(self, t, qi, xi, B_r_CP=zeros3):
        point_dict = self.get_interaction_point(xi)
        _eval = self.kinematics._eval(point_dict, qi, 0)
        if B_r_CP @ B_r_CP == 0.0:
            return _eval[0]

        return _eval[0] + _eval[1] @ B_r_CP

    def r_OP_q(self, t, qi, xi, B_r_CP=zeros3):
        point_dict = self.get_interaction_point(xi)
        _eval, _deval = self.kinematics._eval(point_dict, qi, 1)
        if B_r_CP @ B_r_CP == 0.0:
            return _deval[0]

        r_CP_q = np.einsum("ijk,j->ik", _deval[1], B_r_CP)
        return _deval[0] + r_CP_q

    def A_IB(self, t, qi, xi):
        point_dict = self.get_interaction_point(xi)
        _eval = self.kinematics._eval(point_dict, qi, 0)
        return _eval[1]

    def A_IB_q(self, t, qi, xi):
        point_dict = self.get_interaction_point(xi)
        _eval, _deval = self.kinematics._eval(point_dict, qi, 1)
        return _deval[1]

    # TODO: this belongs to CosseratRod_Kinematics
    @classmethod
    def straight_configuration(cls, nelement, L, r_OP0=zeros3, A_IB0=eye3):
        if cls._parametrization == "Quaternion":
            P = Log_SO3_quat(A_IB0)
            nq_node = 7
        else:
            P = Log_SO3_R9(A_IB0)
            nq_node = 12

        mesh = cls._mesh_kin(None, nelement)
        nnodes = mesh.nnodes

        r_OP = np.zeros((3, nnodes))
        r_OP[0] = np.linspace(0, L, num=nnodes)
        rP = np.zeros((nnodes, nq_node), dtype=float)
        for i in range(nnodes):
            rP[i, :3] = r_OP0 + A_IB0 @ r_OP[:, i]
            rP[i, 3:] = P

        if cls._IGA:
            A = mesh.shape_functions(np.linspace(0, 1, nnodes))[0]
            rP = spsolve(A, rP)
        return rP.reshape(-1)

    @classmethod
    def pose_configuration(cls, nelement, r_OP, A_IB, xi1=1, r_OP0=zeros3, A_IB0=eye3):
        assert callable(r_OP), "r_OP must be callable!"
        assert callable(A_IB), "A_IB must be callable!"

        if cls._parametrization == "Quaternion":
            Log_fct = Log_SO3_quat
            nq_node = 7
        else:
            Log_fct = Log_SO3_R9
            nq_node = 12

        mesh = cls._mesh_kin(None, nelement)
        nnodes = mesh.nnodes
        xis = np.linspace(0, xi1, nnodes)

        # nodal positions and unit quaternions
        rP = np.zeros((nnodes, nq_node))
        for i, xii in enumerate(xis):
            rP[i, :3] = r_OP0 + A_IB0 @ r_OP(xii)
            A_IBi = A_IB0 @ A_IB(xii)
            rP[i, 3:] = Log_fct(A_IBi)

        # check for the right quaternion hemisphere
        for i in range(nnodes - 1):
            inner = rP[i, 3:] @ rP[i + 1, 3:]
            if inner < 0:
                rP[i + 1, 3:] *= -1

        if cls._IGA:
            A = mesh.shape_functions(np.linspace(0, 1, nnodes))[0]
            rP = spsolve(A, rP)
        return rP.reshape(-1)

    @classmethod
    def serret_frenet_configuration(
        cls,
        nelement,
        r_OP,
        r_OP_xi,
        r_OP_xixi,
        xi1,
        alpha=0.0,
        r_OP0=zeros3,
        A_IB0=eye3,
    ):
        """Compute generalized position coordinates for a pre-curved rod along curve r_OP. The cross-section orientations are based on the Serret-Frenet equations and afterwards rotated by alpha."""
        r_OP, r_OP_xi, r_OP_xixi = check_time_derivatives(r_OP, r_OP_xi, r_OP_xixi)
        alpha, _, _ = check_time_derivatives(alpha, None, None)

        def A_IB(xi):
            r_xi = r_OP_xi(xi)
            r_xixi = r_OP_xixi(xi)
            ex = r_xi / np.linalg.norm(r_xi)
            ey = r_xixi - ex * (ex @ r_xixi)
            ey = ey / np.linalg.norm(ey)
            return np.vstack([ex, ey, np.cross(ex, ey)]).T

        return cls.pose_configuration(
            nelement, r_OP, A_IB, xi1, r_OP0=r_OP0, A_IB0=A_IB0
        )

    @classmethod
    def straight_initial_configuration(
        cls,
        nelement,
        L,
        r_OP0=zeros3,
        A_IB0=eye3,
        v_P0=zeros3,
        B_omega_IB0=zeros3,
    ):
        q = cls.straight_configuration(nelement, L, r_OP0, A_IB0)

        mesh = cls._mesh_kin(None, nelement)
        nnodes = mesh.nnodes

        r_OP = np.zeros((nnodes, 3))
        r_OP[:, 0] = np.linspace(0, L, num=nnodes)
        r_OP = r_OP0 + r_OP @ A_IB0.T

        I_omega_IB0 = A_IB0 @ B_omega_IB0
        vO = np.zeros((nnodes, 6), dtype=float)
        vO[:, :3] = v_P0 + np.cross(I_omega_IB0, r_OP - r_OP0)
        vO[:, 3:] = B_omega_IB0

        if cls._IGA:
            A = mesh.shape_functions(np.linspace(0, 1, nnodes))[0]
            vO = spsolve(A, vO)

        return q, vO.reshape(-1)


class CosseratRod_Kinematics(ABC):
    def __init__(self, parent, mesh):
        self.parent = parent
        self.mesh = mesh

    @abstractmethod
    def _eval(point_dict, q, deval=False):
        """if deval==0: returns (r_OP, A_IB),
        if deval==1: returns (r_OP, A_IB), (r_OP_q, A_IB_q),
        if deval==2: returns (r_OP, A_IB), (r_OP_q, A_IB_q), (r_OP_qq, A_IB_qq)"""
        ...


class CosseratRod_Quaternion_R12(CosseratRod_Kinematics):
    def __init__(self, parent, mesh, parametrization):
        super().__init__(parent, mesh)
        self.parametrization = parametrization

        assert parametrization in ["Quaternion", "R12"]
        if parametrization == "Quaternion":
            self.nq_node = 7

            self._A_IB = Exp_SO3_quat
            self._A_IB_P = Exp_SO3_quat_P
            self._T_IB = T_SO3_quat
            self._T_IB_P = T_SO3_quat_P

        else:
            self.nq_node = 12

            self._A_IB = Exp_SO3_R9
            self._A_IB_P = Exp_SO3_R9_R9
            self._T_IB = T_SO3_R9
            self._T_IB_P = T_SO3_R9_R9

        # export and visualization
        self.parent.nodes = self.nodes
        self.parent.nodalFrames = self.nodalFrames
        self.parent.centerline = self.centerline
        self.parent.frames = self.frames

    def _eval(self, point_dict, qi, deval=0):
        N = point_dict["N"]
        qnodes = qi.reshape(point_dict["nnodes"], -1)
        rP = N @ qnodes
        _eval = (rP[:3], self._A_IB(rP[3:]))
        if deval == 0:
            return _eval

        A_IB_P = self._A_IB_P(rP[3:])
        A_IB_q = np.einsum("ijk,kl->ijl", A_IB_P, point_dict["Nq"][3:])

        _deval = (point_dict["Nq"][:3], A_IB_q)
        return _eval, _deval

    def _eval_vec(self, N, q):
        qnodes = q.reshape(self.parent.nnodes, -1)
        rP = N @ qnodes
        return rP[:, :3], self._A_IB(rP[:, 3:])

    def _eval_internal_vec(self, N, N_xi, q, deval=0):
        qbar_nodes = q.reshape(self.parent.nnodes, -1)
        P_IB = N @ qbar_nodes[:, 3:]
        qbar_xi = N_xi @ qbar_nodes

        A_IB = self._A_IB(P_IB)
        T = self._T_IB(P_IB)

        B_gamma_bar = np.einsum("ijk,ij->ik", A_IB, qbar_xi[:, :3])
        B_kappa_bar = np.einsum("ijk,ik->ij", T, qbar_xi[:, 3:])
        _eval = (A_IB, B_gamma_bar, B_kappa_bar)
        if deval == 0:
            return _eval

        # using my magic property
        B_gamma_bar_P = np.cross(B_gamma_bar[:, :, None], T, axisa=1, axisb=1, axisc=1)

        T_IB_P = self._T_IB_P(P_IB)
        B_kappa_bar_P = np.einsum("ijkl,ik->ijl", T_IB_P, qbar_xi[:, 3:])
        _deval = (T, B_gamma_bar_P, B_kappa_bar_P)
        if deval == 1:
            return _eval, _deval

        B_gamma_bar_rP = ...
        _ddeval = (B_gamma_bar_rP, B_gamma_bar_PP, B_kappa_bar_PP)
        return _eval, _deval, _ddeval

    # TODO: which class/where to generalize?
    ############################
    # export of centerline nodes
    ############################
    def nodes(self, qsystem):
        """Returns nodal position coordinates"""
        qbody = qsystem[self.parent.qDOF]
        qnodesT = qbody.reshape(-1, self.parent.nnodes, order="F")
        return qnodesT[:3]

    def nodalFrames(self, qsystem, elementwise=False):
        """Returns nodal positions and nodal directors.
        If elementwise==True : returned arrays are each of shape [nnodes, 3]
        If elementwise==False : returned arrays are each of shape [nelements, nnodes_per_element, 3]
        """
        qbody = qsystem[self.qDOF]
        if elementwise:
            raise NotImplementedError
        else:
            qnodes = qbody.reshape(self.nnodes, -1)
            A_IB = self._A_IB(qnodes[:, 3:])
            return qnodes[:, :3], A_IB[:, :, 0], A_IB[:, :, 1], A_IB[:, :, 2]

    def centerline(self, q, num=100):
        xis = np.linspace(0, 1, num)
        els = self.parent.element_number(xis)
        N = self.parent.N(xis, els)[0]
        q_body = q[self.parent.qDOF]
        q_nodes = q_body.reshape(self.parent.nnodes, -1)
        r_OC = N @ q_nodes[:, :3]
        return r_OC.T

    def frames(self, q, num=10):
        xis = np.linspace(0, 1, num)
        els = self.parent.element_number(xis)
        N = self.parent.N(xis, els)[0]
        r, A_IB = self._eval_vec(N, q[self.parent.qDOF])
        return r.T, A_IB[:, :, 0].T, A_IB[:, :, 1].T, A_IB[:, :, 2].T

    def _export_nodes(self, solution):
        # TODO: allow for higher resolution than self.nnodes
        num_bones = self.parent.nnodes
        data = np.empty((len(solution.t), num_bones, 7), dtype=float)
        for i in range(len(solution.t)):
            data[i] = solution.q[i, self.parent.qDOF].reshape(self.parent.nnodes, -1)

        return np.linspace(0, 1, num_bones), data

    def _export_nodes_modes(self, solution):
        # TODO: allow for higher resolution than self.nnodes
        num_bones = self.parent.nnodes
        data = solution.q[self.parent.qDOF].reshape(self.parent.nnodes, -1)
        delta = solution.Delta_z[self.parent.uDOF].reshape(
            self.parent.nnodes, -1, len(solution.omegas)
        )

        return np.linspace(0, 1, num_bones), data, delta
