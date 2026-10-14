from abc import ABC
import numpy as np
from warnings import warn

from cardillo.math.algebra import ax2skew, ax2skew_a, skew2ax
from cardillo.math.approx_fprime import approx_fprime
from scipy.sparse import csr_array

from .cosseratRod_MatrixUtils import dense_block_diag

zeros3 = np.zeros(3, dtype=float)
eye3 = np.eye(3, dtype=float)


class CosseratRod_Velocity(ABC):
    def __init__(self, parent, mesh, quadrature_ext):
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

        # quadrature
        quadrature_ext = mesh.quadrature(*quadrature_ext, 1)
        self.nquadrature_ext_total = quadrature_ext["nquadrature_total"]
        self.qp_ext_vec = quadrature_ext["qp"]
        self.qw_ext_vec = quadrature_ext["qw"]
        self.qels_ext_vec = quadrature_ext["els"]
        self.N_ext, self.N_xi_ext = quadrature_ext["N"]

    def set_distributed_load(self, distributed_load):
        assert (
            len(distributed_load) == 2
        ), "Line distributed forces must be a list of length 2 (force and moment)."
        if (distributed_load[0] == None) and (distributed_load[1] == None):
            self.include_f_ext = False
        else:
            self.include_f_ext = True

            zeros_ext = np.zeros((len(self.qp_ext_vec), 3))
            self.distributed_load = distributed_load
            if self.distributed_load[0] is None:
                self.distributed_load[0] = lambda t, xis: zeros_ext
            if self.distributed_load[1] is None:
                self.distributed_load[1] = lambda t, xis: zeros_ext

    def set_reference_strains(self, Q):
        # external virtual work contributions
        _, B_gamma_bar, _ = self.parent.kinematics._eval_internal_vec(
            self.N_ext, self.N_xi_ext, Q
        )
        J_ext_vec = np.linalg.norm(B_gamma_bar, axis=1)
        self.weights_ext = J_ext_vec * self.qw_ext_vec


class CosseratRod_PG_IB(CosseratRod_Velocity):
    def __init__(self, parent, mesh, quadrature_ext):
        super().__init__(parent, mesh, quadrature_ext)
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

        A_IB = self.parent.A_IB(t, qi, xi)
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

    #################
    # external load #
    #################
    def E_pot_ext(self, t, q):
        # potential of the distributed force, distributed moments have none
        r_OC, _ = self.parent.kinematics._eval_vec(self.N_ext, q)
        b_qp = self.distributed_load[0](t, self.qp_ext_vec)
        return -np.einsum("ij,ij", r_OC, b_qp * self.weights_ext[:, None])

    def f_ext(self, t, q, u):
        b_qp = self.distributed_load[0](t, self.qp_ext_vec)
        B_c_qp = self.distributed_load[1](t, self.qp_ext_vec)

        f_ext = np.empty((self.parent.nnodes, 6))
        f_ext[:, :3] = self.N_ext.T @ (b_qp * self.weights_ext[:, None])
        f_ext[:, 3:] = self.N_ext.T @ (B_c_qp * self.weights_ext[:, None])
        return f_ext.reshape(-1)


def _B_J_R(A_IB, A_IB_q):
    """Body-fixed angular velocity Jacobian, i.e., A_IB.T @ A_IB_q = skew(B_J_R)."""
    return skew2ax(np.einsum("ji,jkc->cik", A_IB, A_IB_q)).T


def _B_J_R_q(A_IB, A_IB_q, A_IB_qq):
    """Derivative of _B_J_R w.r.t. q."""
    return skew2ax(
        np.einsum("jid,jkc->cdik", A_IB_q, A_IB_q)
        + np.einsum("ji,jkcd->cdik", A_IB, A_IB_qq)
    ).transpose(2, 0, 1)


class _BubnovGalerkinLoad:
    """Distributed loads for the Bubnov-Galerkin formulation, i.e., the virtual
    work of b @ delta r + B_c @ delta phi with delta phi = Psi_q @ delta q,
    which results in the generalized forces f_BG = _f_ext_BG(t, q)."""

    def set_distributed_load(self, distributed_load):
        # distributed moments and the SE3 interpolation of the centerline make
        # f_ext dependent on q
        self.include_distributed_moment = distributed_load[1] is not None
        self.f_ext_depends_on_q = (
            self.include_distributed_moment or self.parent._parametrization == "SE3"
        )
        super().set_distributed_load(distributed_load)

    def E_pot_ext(self, t, q):
        # potential of the distributed force, distributed moments have none
        r_OC, _ = self.parent.kinematics._eval_vec(self.N_ext, q)
        b_qp = self.distributed_load[0](t, self.qp_ext_vec)
        return -np.einsum("ij,ij", r_OC, b_qp * self.weights_ext[:, None])

    def _f_ext_BG(self, t, q):
        b_qp = self.distributed_load[0](t, self.qp_ext_vec)
        B_c_qp = self.distributed_load[1](t, self.qp_ext_vec)
        _eval, _deval = self.parent.kinematics._eval_internal_vec(
            self.N_ext, self.N_xi_ext, q, deval=1
        )
        _, A_IB_qbar, r_OC_qbar = _deval

        # f_qp[N/N_xi, qpi, qDOF]: b @ r_OC_q (+ B_c @ Psi_q)
        f_qp = np.einsum("apik,pi->apk", r_OC_qbar, b_qp)
        if self.include_distributed_moment:
            # Psi_qbar[N/N_xi, qpi, qDOF, 3]
            Psi_qbar = skew2ax(np.einsum("pji,apjkc->apcik", _eval[0], A_IB_qbar))
            f_qp += np.einsum("apci,pi->apc", Psi_qbar, B_c_qp)
        f_qp *= self.weights_ext[None, :, None]
        f_BG = self.N_ext.T @ f_qp[0] + self.N_xi_ext.T @ f_qp[1]
        return f_BG.reshape(-1)

    def f_ext_q(self, t, q, u):
        if not self.f_ext_depends_on_q:
            return csr_array((self.parent.nu, self.parent.nq))
        # TODO: analytic derivative
        return approx_fprime(q, lambda q_: self.f_ext(t, q_, u))


class CosseratRod_DBG(_BubnovGalerkinLoad, CosseratRod_PG_IB):
    """Bubnov-Galerkin formulation with discrete nullspace projection, i.e.,
    the velocities are projected with the nodal kinematic equation
    q_dot = B(q) @ u. At the nodes, this coincides with CosseratRod_PG_IB."""

    def _B(self, qi, nnodes):
        """Local projection B (nqi, nui) of the kinematic equation q_dot = B @ u."""
        return dense_block_diag(self.parent.kin_eq.B_blocks(qi))

    def _B_q(self, qi, nnodes):
        """Derivative of _B w.r.t. qi, i.e., B_q (nqi, nui, nqi)."""
        return dense_block_diag(self.parent.kin_eq.B_q_blocks(qi))

    def _nnodes(self, xi):
        return self.parent.get_interaction_point(xi)["nnodes"]

    def f_ext(self, t, q, u):
        return self.parent.kin_eq.BT_f(q, self._f_ext_BG(t, q))

    def KN_f_ext(self, t, q, u):
        if not self.f_ext_depends_on_q:
            return None, None
        # K + N = -f_ext_q @ B, split into symmetric and skew-symmetric part
        f_ext_q_u = -self.f_ext_q(t, q, u) @ self.parent.q_dot_u(t, q)
        return 0.5 * (f_ext_q_u + f_ext_q_u.T), 0.5 * (f_ext_q_u - f_ext_q_u.T)

    ################################
    # velocity and its derivatives #
    ################################
    def J_P(self, t, qi, xi, B_r_CP=zeros3):
        if (nnodes := self._nnodes(xi)) == 1:
            return super().J_P(t, qi, xi, B_r_CP)
        return self.parent.r_OP_q(t, qi, xi, B_r_CP) @ self._B(qi, nnodes)

    def J_P_q(self, t, qi, xi, B_r_CP=zeros3):
        if (nnodes := self._nnodes(xi)) == 1:
            return super().J_P_q(t, qi, xi, B_r_CP)
        r_OP_q = self.parent.r_OP_q(t, qi, xi, B_r_CP)
        r_OP_qq = self.parent._r_OP_qq(t, qi, xi, B_r_CP)
        return np.einsum("ikl,km->iml", r_OP_qq, self._B(qi, nnodes)) + np.einsum(
            "ik,kml->iml", r_OP_q, self._B_q(qi, nnodes)
        )

    def J2_P(self, t, qi, xi, B_r_CP=zeros3):
        if self._nnodes(xi) == 1:
            return super().J2_P(t, qi, xi, B_r_CP)
        raise NotImplementedError("J2_P is only implemented at the nodes.")

    def v_P(self, t, qi, ui, xi, B_r_CP=zeros3):
        if self._nnodes(xi) == 1:
            return super().v_P(t, qi, ui, xi, B_r_CP)
        return self.J_P(t, qi, xi, B_r_CP) @ ui

    def v_P_q(self, t, qi, ui, xi, B_r_CP=zeros3):
        if self._nnodes(xi) == 1:
            return super().v_P_q(t, qi, ui, xi, B_r_CP)
        return np.einsum("ijk,j->ik", self.J_P_q(t, qi, xi, B_r_CP), ui)

    def a_P(self, t, qi, ui, ui_dot, xi, B_r_CP=zeros3):
        if (nnodes := self._nnodes(xi)) == 1:
            return super().a_P(t, qi, ui, ui_dot, xi, B_r_CP)
        qi_dot = self._B(qi, nnodes) @ ui
        return self.J_P(t, qi, xi, B_r_CP) @ ui_dot + np.einsum(
            "ijk,j,k->i", self.J_P_q(t, qi, xi, B_r_CP), ui, qi_dot
        )

    def a_P_q(self, t, qi, ui, ui_dot, xi, B_r_CP=zeros3):
        if self._nnodes(xi) == 1:
            return super().a_P_q(t, qi, ui, ui_dot, xi, B_r_CP)
        return approx_fprime(qi, lambda qi_: self.a_P(t, qi_, ui, ui_dot, xi, B_r_CP))

    def a_P_u(self, t, qi, ui, ui_dot, xi, B_r_CP=zeros3):
        if self._nnodes(xi) == 1:
            return super().a_P_u(t, qi, ui, ui_dot, xi, B_r_CP)
        return approx_fprime(ui, lambda ui_: self.a_P(t, qi, ui_, ui_dot, xi, B_r_CP))

    ########################################
    # angular velocity and its derivatives #
    ########################################
    def B_J_R(self, t, qi, xi):
        if (nnodes := self._nnodes(xi)) == 1:
            return super().B_J_R(t, qi, xi)
        A_IB, A_IB_q = self.parent.A_IB(t, qi, xi), self.parent.A_IB_q(t, qi, xi)
        return _B_J_R(A_IB, A_IB_q) @ self._B(qi, nnodes)

    def B_J_R_q(self, t, qi, xi):
        if (nnodes := self._nnodes(xi)) == 1:
            return super().B_J_R_q(t, qi, xi)
        A_IB, A_IB_q = self.parent.A_IB(t, qi, xi), self.parent.A_IB_q(t, qi, xi)
        Psi_q = _B_J_R(A_IB, A_IB_q)
        Psi_qq = _B_J_R_q(A_IB, A_IB_q, self.parent._A_IB_qq(t, qi, xi))
        return np.einsum("ikl,km->iml", Psi_qq, self._B(qi, nnodes)) + np.einsum(
            "ik,kml->iml", Psi_q, self._B_q(qi, nnodes)
        )

    def B_J2_R(self, t, qi, xi):
        if self._nnodes(xi) == 1:
            return super().B_J2_R(t, qi, xi)
        raise NotImplementedError("B_J2_R is only implemented at the nodes.")

    def B_Omega(self, t, qi, ui, xi):
        if self._nnodes(xi) == 1:
            return super().B_Omega(t, qi, ui, xi)
        return self.B_J_R(t, qi, xi) @ ui

    def B_Omega_q(self, t, qi, ui, xi):
        if self._nnodes(xi) == 1:
            return super().B_Omega_q(t, qi, ui, xi)
        return np.einsum("ijk,j->ik", self.B_J_R_q(t, qi, xi), ui)

    def B_Psi(self, t, qi, ui, ui_dot, xi):
        if (nnodes := self._nnodes(xi)) == 1:
            return super().B_Psi(t, qi, ui, ui_dot, xi)
        qi_dot = self._B(qi, nnodes) @ ui
        return self.B_J_R(t, qi, xi) @ ui_dot + np.einsum(
            "ijk,j,k->i", self.B_J_R_q(t, qi, xi), ui, qi_dot
        )

    def B_Psi_q(self, t, qi, ui, ui_dot, xi):
        if self._nnodes(xi) == 1:
            return super().B_Psi_q(t, qi, ui, ui_dot, xi)
        return approx_fprime(qi, lambda qi_: self.B_Psi(t, qi_, ui, ui_dot, xi))

    def B_Psi_u(self, t, qi, ui, ui_dot, xi):
        if self._nnodes(xi) == 1:
            return super().B_Psi_u(t, qi, ui, ui_dot, xi)
        return approx_fprime(ui, lambda ui_: self.B_Psi(t, qi, ui_, ui_dot, xi))


class CosseratRod_BG(_BubnovGalerkinLoad, CosseratRod_Velocity):
    """Bubnov-Galerkin formulation, i.e., u = q_dot. All point quantities follow
    from the partial derivatives of r_OP and A_IB w.r.t. q."""

    def __init__(self, parent, mesh, quadrature_ext):
        super().__init__(parent, mesh, quadrature_ext)
        self.nu_node = self.parent.kinematics.nq_node

    #################
    # external load #
    #################
    def f_ext(self, t, q, u):
        return self._f_ext_BG(t, q)

    def KN_f_ext(self, t, q, u):
        if not self.f_ext_depends_on_q:
            return None, None
        # u = q_dot: K + N = -f_ext_q, split into symmetric and skew-symmetric part
        f_ext_q_u = -self.f_ext_q(t, q, u)
        return 0.5 * (f_ext_q_u + f_ext_q_u.T), 0.5 * (f_ext_q_u - f_ext_q_u.T)

    ################################
    # velocity and its derivatives #
    ################################
    def J_P(self, t, qi, xi, B_r_CP=zeros3):
        return self.parent.r_OP_q(t, qi, xi, B_r_CP)

    def J_P_q(self, t, qi, xi, B_r_CP=zeros3):
        return self.parent._r_OP_qq(t, qi, xi, B_r_CP)

    def J2_P(self, t, qi, xi, B_r_CP=zeros3):
        raise NotImplementedError("J2_P is not implemented for BG.")

    def v_P(self, t, qi, ui, xi, B_r_CP=zeros3):
        return self.J_P(t, qi, xi, B_r_CP) @ ui

    def v_P_q(self, t, qi, ui, xi, B_r_CP=zeros3):
        return np.einsum("ijk,j->ik", self.J_P_q(t, qi, xi, B_r_CP), ui)

    def a_P(self, t, qi, ui, ui_dot, xi, B_r_CP=zeros3):
        r_OP_q = self.parent.r_OP_q(t, qi, xi, B_r_CP)
        r_OP_qq = self.parent._r_OP_qq(t, qi, xi, B_r_CP)
        return r_OP_q @ ui_dot + np.einsum("ijk,j,k->i", r_OP_qq, ui, ui)

    def a_P_q(self, t, qi, ui, ui_dot, xi, B_r_CP=zeros3):
        # TODO: third derivatives of A_IB
        return approx_fprime(qi, lambda qi_: self.a_P(t, qi_, ui, ui_dot, xi, B_r_CP))

    def a_P_u(self, t, qi, ui, ui_dot, xi, B_r_CP=zeros3):
        # r_OP_qq is symmetric
        return 2.0 * np.einsum("ijk,k->ij", self.J_P_q(t, qi, xi, B_r_CP), ui)

    ########################################
    # angular velocity and its derivatives #
    ########################################
    def B_J_R(self, t, qi, xi):
        return _B_J_R(self.parent.A_IB(t, qi, xi), self.parent.A_IB_q(t, qi, xi))

    def B_J_R_q(self, t, qi, xi):
        A_IB, A_IB_q = self.parent.A_IB(t, qi, xi), self.parent.A_IB_q(t, qi, xi)
        return _B_J_R_q(A_IB, A_IB_q, self.parent._A_IB_qq(t, qi, xi))

    def B_J2_R(self, t, qi, xi):
        raise NotImplementedError("B_J2_R is not implemented for BG.")

    def B_Omega(self, t, qi, ui, xi):
        return self.B_J_R(t, qi, xi) @ ui

    def B_Omega_q(self, t, qi, ui, xi):
        return np.einsum("ijk,j->ik", self.B_J_R_q(t, qi, xi), ui)

    def B_Psi(self, t, qi, ui, ui_dot, xi):
        A_IB, A_IB_q = self.parent.A_IB(t, qi, xi), self.parent.A_IB_q(t, qi, xi)
        Psi_q = _B_J_R(A_IB, A_IB_q)
        Psi_qq = _B_J_R_q(A_IB, A_IB_q, self.parent._A_IB_qq(t, qi, xi))
        return Psi_q @ ui_dot + np.einsum("ijk,j,k->i", Psi_qq, ui, ui)

    def B_Psi_q(self, t, qi, ui, ui_dot, xi):
        # TODO: third derivatives of A_IB
        return approx_fprime(qi, lambda qi_: self.B_Psi(t, qi_, ui, ui_dot, xi))

    def B_Psi_u(self, t, qi, ui, ui_dot, xi):
        Psi_qq = self.B_J_R_q(t, qi, xi)
        return np.einsum("ijk,k->ij", Psi_qq, ui) + np.einsum("ijk,j->ik", Psi_qq, ui)
