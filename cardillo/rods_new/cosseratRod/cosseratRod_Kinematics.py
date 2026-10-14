from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse.linalg import spsolve

from cardillo.math.rotations import (
    Log_SO3_quat,
    Exp_SO3_quat,
    Exp_SO3_quat_P,
    Exp_SO3_quat_PP,
    T_SO3_quat,
    T_SO3_quat_P,
    T_SO3_quat_PP,
    T_SO3_inv_quat,
    Log_SO3_R9,
    Exp_SO3_R9,
    Exp_SO3_R9_R9,
    T_SO3_R9,
    T_SO3_R9_R9,
    SE3,
    Exp_SE3,
    SE3inv,
    Log_SE3,
    T_SE3,
    T_SE3_inv,
    Ad_SE3,
)
from cardillo.math.algebra import ax2skew, ax2skew_a
from cardillo.utility.check_time_derivatives import check_time_derivatives

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
        if cls._parametrization in ["Quaternion", "SE3"]:
            P = Log_SO3_quat(A_IB0)
            nq_node = 7

        else:
            P = Log_SO3_R9(A_IB0)
            nq_node = 12

        mesh = cls._mesh_kin(None, nelement)
        nnodes = mesh.nnodes

        if cls._parametrization == "SE3":
            assert mesh.polynomial_degree == 1

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

        if cls._parametrization in ["Quaternion", "SE3"]:
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
        v_C = v_P0 + np.cross(I_omega_IB0, r_OP - r_OP0)
        if cls._projection:
            # nodal velocities v_C and B_omega_IB
            vO = np.zeros((nnodes, 6), dtype=float)
            vO[:, :3] = v_C
            vO[:, 3:] = B_omega_IB0
        else:
            # Bubnov-Galerkin, i.e., u = q_dot
            qnodes = q.reshape(nnodes, -1)
            vO = np.zeros_like(qnodes)
            vO[:, :3] = v_C
            if cls._parametrization == "R12":
                A_IB0_dot = A_IB0 @ ax2skew(B_omega_IB0)
                vO[:, 3:] = A_IB0_dot.reshape(-1, order="F")
            else:
                vO[:, 3:] = T_SO3_inv_quat(qnodes[:, 3:]) @ B_omega_IB0

        if cls._IGA:
            A = mesh.shape_functions(np.linspace(0, 1, nnodes))[0]
            vO = spsolve(A, vO)

        return q, vO.reshape(-1)


class CosseratRod_Kinematics(ABC):
    def __init__(self, parent, mesh):
        self.parent = parent
        self.mesh = mesh

        self.parent.step_callback = self.step_callback

        # export and visualization
        self.parent.nodes = self.nodes
        self.parent.nodalFrames = self.nodalFrames
        self.parent.centerline = self.centerline
        self.parent.frames = self.frames

    @abstractmethod
    def _eval(self, point_dict, q, deval=False):
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
            self._A_IB_PP = Exp_SO3_quat_PP
            self._T_IB = T_SO3_quat
            self._T_IB_P = T_SO3_quat_P
            self._T_IB_PP = T_SO3_quat_PP

        else:
            self.nq_node = 12

            self._A_IB = Exp_SO3_R9
            self._A_IB_P = Exp_SO3_R9_R9
            self._T_IB = T_SO3_R9
            self._T_IB_P = T_SO3_R9_R9
            # A_IB and T_IB are linear in R9
            self._A_IB_PP = lambda R9: np.zeros(np.shape(R9)[:-1] + (3, 3, 9, 9))
            self._T_IB_PP = lambda R9: np.zeros(np.shape(R9)[:-1] + (3, 9, 9, 9))

    def step_callback(self, t, q, u):
        """Restore orthonormality of the nodal frame parametrization after
        each time step (unit quaternion for "Quaternion", orthonormal
        directors for "R12")."""
        qnodes = q.reshape(self.parent.nnodes, -1)
        if self.parametrization == "Quaternion":
            qnodes[:, 3:] /= np.linalg.norm(qnodes[:, 3:], axis=1)[:, None]
        else:
            # nearest orthogonal matrix to [d1 | d2 | d3] in Frobenius norm
            A_IB = np.stack([qnodes[:, 3:6], qnodes[:, 6:9], qnodes[:, 9:12]], axis=-1)
            U, _, Vt = np.linalg.svd(A_IB)
            A_IB = U @ Vt
            qnodes[:, 3:6] = A_IB[:, :, 0]
            qnodes[:, 6:9] = A_IB[:, :, 1]
            qnodes[:, 9:12] = A_IB[:, :, 2]

        return qnodes.reshape(-1), u

    def _eval(self, point_dict, qi, deval=0):
        N = point_dict["N"]
        qnodes = qi.reshape(point_dict["nnodes"], -1)
        rP = N @ qnodes
        _eval = (rP[:3], self._A_IB(rP[3:]))
        if deval == 0:
            return _eval

        Nq_P = point_dict["Nq"][3:]
        A_IB_P = self._A_IB_P(rP[3:])
        A_IB_q = np.einsum("ijk,kl->ijl", A_IB_P, Nq_P)

        _deval = (point_dict["Nq"][:3], A_IB_q)
        if deval == 1:
            return _eval, _deval

        nqi = len(qi)
        A_IB_qq = np.einsum("ijab,ak,bl->ijkl", self._A_IB_PP(rP[3:]), Nq_P, Nq_P)
        _ddeval = (np.zeros((3, nqi, nqi)), A_IB_qq)
        return _eval, _deval, _ddeval

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

        A_IB_P = self._A_IB_P(P_IB)
        T_IB_P = self._T_IB_P(P_IB)

        # derivatives w.r.t. qbar[N/N_xi, qpi, ..., qDOF], i.e., to be
        # multiplied with N (index 0) or N_xi (index 1) of the respective node
        nqp = N.shape[0]
        epsilon_bar_qbar = np.zeros((2, nqp, 6, self.nq_node))
        epsilon_bar_qbar[0, :, :3, 3:] = np.einsum(
            "ijkl,ij->ikl", A_IB_P, qbar_xi[:, :3]
        )
        epsilon_bar_qbar[0, :, 3:, 3:] = np.einsum(
            "ijkl,ik->ijl", T_IB_P, qbar_xi[:, 3:]
        )
        epsilon_bar_qbar[1, :, :3, :3] = A_IB.transpose(0, 2, 1)
        epsilon_bar_qbar[1, :, 3:, 3:] = T

        A_IB_qbar = np.zeros((2, nqp, 3, 3, self.nq_node))
        A_IB_qbar[0, ..., 3:] = A_IB_P

        _deval = (epsilon_bar_qbar, A_IB_qbar)
        if deval == 1:
            return _eval, _deval

        # second derivatives
        # epsilon_bar_qbarqbar[N/N_xi, N/N_xi, qpi, epsilonDOF, qDOF, qDOF]
        epsilon_bar_qbarqbar = np.zeros((2, 2, nqp, 6, self.nq_node, self.nq_node))
        # B_gamma_bar = A_IB.T @ r_xi
        epsilon_bar_qbarqbar[0, 0, :, :3, 3:, 3:] = np.einsum(
            "ijkab,ij->ikab", self._A_IB_PP(P_IB), qbar_xi[:, :3]
        )
        epsilon_bar_qbarqbar[0, 1, :, :3, 3:, :3] = A_IB_P.transpose(0, 2, 3, 1)
        epsilon_bar_qbarqbar[1, 0, :, :3, :3, 3:] = A_IB_P.transpose(0, 2, 1, 3)
        # B_kappa_bar = T_IB @ P_xi
        epsilon_bar_qbarqbar[0, 0, :, 3:, 3:, 3:] = np.einsum(
            "ijkab,ik->ijab", self._T_IB_PP(P_IB), qbar_xi[:, 3:]
        )
        epsilon_bar_qbarqbar[0, 1, :, 3:, 3:, 3:] = T_IB_P.transpose(0, 1, 3, 2)
        epsilon_bar_qbarqbar[1, 0, :, 3:, 3:, 3:] = T_IB_P

        return _eval, _deval, (epsilon_bar_qbarqbar,)

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


class CosseratRod_SE3(CosseratRod_Quaternion_R12):
    def __init__(self, parent, mesh, parametrization):
        assert (
            parametrization == "SE3"
        ), "We should never be here without parametrization=='SE3'!"
        assert (
            mesh.polynomial_degree == 1
        ), "SE3 interpolation requires linear elements!"
        super().__init__(parent, mesh, "Quaternion")

    def _eval(self, point_dict, qi, deval=0):
        assert point_dict["nnodes"] in [1, 2]
        q_nodes = qi.reshape(point_dict["nnodes"], 7)
        A_IB_nodes = Exp_SO3_quat(q_nodes[:, 3:])
        H_IB_nodes = SE3(A_IB_nodes, q_nodes[:, :3])
        if point_dict["nnodes"] == 1:
            H_IB = H_IB_nodes[0]
        else:
            N_1 = point_dict["N"][1]
            h = Log_SE3(SE3inv(H_IB_nodes[0]) @ H_IB_nodes[1])
            H_B0B = Exp_SE3(N_1 * h)
            H_IB = H_IB_nodes[0] @ H_B0B

        A_IB = H_IB[:3, :3]
        _eval = (H_IB[:3, 3], A_IB)
        if deval == 0:
            return _eval

        # body-fixed derivative s_q, i.e., H_IB^-1 @ H_IB_q = hat(s_q)
        s_q_nodes, _, h_q0, h_q1 = self._element_derivatives(
            q_nodes, A_IB_nodes, H_IB_nodes
        )
        if point_dict["nnodes"] == 1:
            s_q = s_q_nodes[0]
        else:
            T_N_1 = N_1 * T_SE3(N_1 * h)
            s_q = np.hstack(
                [
                    Ad_SE3(SE3inv(H_B0B)) @ s_q_nodes[0] + T_N_1 @ h_q0[0],
                    T_N_1 @ h_q1[0],
                ]
            )

        # r_OP_q = A_IB @ s_q[:3], A_IB_q = A_IB @ skew(s_q[3:])
        r_OP_q = A_IB @ s_q[:3]
        A_IB_q = np.einsum("jl,lkm,mc->jkc", A_IB, ax2skew_a(), s_q[3:])
        return _eval, (r_OP_q, A_IB_q)

    @staticmethod
    def _element_derivatives(q_nodes, A_IB_nodes, H_IB_nodes):
        """Body-fixed derivatives of the nodal frames s_q_i, i.e.,
        H_IB_i^-1 @ H_IB_i_q_i = hat(s_q_i),
        and derivatives of the element twists h = Log(H_0^-1 @ H_1) w.r.t. the
        coordinates of the first (h_q0) and second (h_q1) node of each element."""
        s_q_nodes = np.zeros((len(q_nodes), 6, 7))
        s_q_nodes[:, :3, :3] = A_IB_nodes.transpose(0, 2, 1)
        s_q_nodes[:, 3:, 3:] = T_SO3_quat(q_nodes[:, 3:])

        # h_q = T_SE3(h)^-1 @ (s_q1 - Ad(H_01^-1) @ s_q0)
        H_01 = SE3inv(H_IB_nodes[:-1]) @ H_IB_nodes[1:]
        h_element = Log_SE3(H_01)
        T_inv_h = T_SE3_inv(h_element)
        h_q0 = -T_inv_h @ Ad_SE3(SE3inv(H_01)) @ s_q_nodes[:-1]
        h_q1 = T_inv_h @ s_q_nodes[1:]
        return s_q_nodes, h_element, h_q0, h_q1

    def els_from_N(self, N):
        n_qp = N.shape[0]
        els = np.zeros(n_qp, dtype=int)
        for qp in range(n_qp):
            el = np.where(N[qp].toarray())[0][0]  # first time with no 0
            els[qp] = min(el, self.parent.nelement - 1)  # case xi=1.0
        return els

    def _eval_vec(self, N, q):
        els = self.els_from_N(N)
        q_nodes = q.reshape(-1, 7)

        # evaluate at nodes and get elements twist
        A_IB_nodes = Exp_SO3_quat(q_nodes[:, 3:])
        H_IB_nodes = SE3(A_IB_nodes, q_nodes[:, :3])
        H_IB_nodes_inv = SE3inv(H_IB_nodes[:-1])  # no need to invert the last
        h_element = Log_SE3(H_IB_nodes_inv @ H_IB_nodes[1:])

        # map to the quadrature points
        N_linear = N[np.arange(N.shape[0]), els + 1]
        h_element_qp = h_element[els]
        H_IB0_qp = H_IB_nodes[els]
        H_B0B_qp = Exp_SE3(N_linear[:, None] * h_element_qp)

        # compose
        H_IB_qp = H_IB0_qp @ H_B0B_qp
        return H_IB_qp[:, :3, 3], H_IB_qp[:, :3, :3]

    def _eval_internal_vec(self, N, N_xi, q, deval=0):
        els = self.els_from_N(N)
        q_nodes = q.reshape(-1, 7)

        # evaluate at nodes and get element twist
        A_IB_nodes = Exp_SO3_quat(q_nodes[:, 3:])
        H_IB_nodes = SE3(A_IB_nodes, q_nodes[:, :3])
        H_IB_nodes_inv = SE3inv(H_IB_nodes[:-1])  # no need to invert the last
        h_element = Log_SE3(H_IB_nodes_inv @ H_IB_nodes[1:])

        # map to the quadrature points
        arange = np.arange(N.shape[0])
        N_linear = N[arange, els + 1]
        N_linear_xi = N_xi[arange, els + 1]
        h_element_qp = h_element[els]
        H_IB0_qp = H_IB_nodes[els]
        H_B0B_qp = Exp_SE3(N_linear[:, None] * h_element_qp)

        # compose
        H_IB_qp = H_IB0_qp @ H_B0B_qp
        epsilon_bar_qp = N_linear_xi[:, None] * h_element_qp

        A_IB_qp = H_IB_qp[:, :3, :3]
        _eval = (A_IB_qp, epsilon_bar_qp[:, :3], epsilon_bar_qp[:, 3:])
        if deval == 0:
            return _eval

        s_q_nodes, _, h_q0, h_q1 = self._element_derivatives(
            q_nodes, A_IB_nodes, H_IB_nodes
        )

        # H_IB = H_IB0 @ Exp(N_1 h):
        # s_q = Ad(Exp(N_1 h)^-1) @ s_q0 + N_1 T_SE3(N_1 h) @ h_q
        h_q0_qp = h_q0[els]
        h_q1_qp = h_q1[els]
        T_qp = N_linear[:, None, None] * T_SE3(N_linear[:, None] * h_element_qp)
        s_q0 = Ad_SE3(SE3inv(H_B0B_qp)) @ s_q_nodes[els] + T_qp @ h_q0_qp
        s_q1 = T_qp @ h_q1_qp

        # derivatives w.r.t. the nodes of the element; A_IB_q = A_IB @ skew(s_q[3:])
        epsilon_bar_q0 = N_linear_xi[:, None, None] * h_q0_qp
        epsilon_bar_q1 = N_linear_xi[:, None, None] * h_q1_qp
        A_IB_q0 = np.einsum("ijl,lkm,imc->ijkc", A_IB_qp, ax2skew_a(), s_q0[:, 3:])
        A_IB_q1 = np.einsum("ijl,lkm,imc->ijkc", A_IB_qp, ax2skew_a(), s_q1[:, 3:])

        # express as derivatives w.r.t. qbar[N/N_xi] such that
        # d_q0 = X N_0 + Y N_0_xi and d_q1 = X N_1 + Y N_1_xi
        N_0 = np.asarray(N[arange, els]).reshape(-1)
        N_0_xi = np.asarray(N_xi[arange, els]).reshape(-1)
        det = N_0 * N_linear_xi - N_linear * N_0_xi

        def to_qbar(d_q0, d_q1):
            expand = (slice(None),) + (None,) * (d_q0.ndim - 1)
            X = (N_linear_xi[expand] * d_q0 - N_0_xi[expand] * d_q1) / det[expand]
            Y = (N_0[expand] * d_q1 - N_linear[expand] * d_q0) / det[expand]
            return np.stack([X, Y])

        _deval = (
            to_qbar(epsilon_bar_q0, epsilon_bar_q1),
            to_qbar(A_IB_q0, A_IB_q1),
        )
        return _eval, _deval

    # TODO: which class/where to generalize?
    ############################
    # export of centerline nodes
    ############################
    def centerline(self, q, num=100):
        xis = np.linspace(0, 1, num)

        r_OCs = np.zeros((3, num))
        for i in range(num):
            xi = xis[i]
            qDOF = self.parent.local_qDOF_P(xi)
            r_OCs[:, i] = self.parent.r_OP(0.0, q[qDOF], xi)

        return r_OCs
