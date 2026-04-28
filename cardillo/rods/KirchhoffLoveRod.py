import numpy as np
from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey
import sparse
from scipy.sparse import (
    bsr_array,
    csr_array,
    eye_array,
)
from scipy.sparse.linalg import spsolve
from warnings import warn

from cardillo.math.algebra import norm, cross3, ax2skew
from cardillo.math.approx_fprime import approx_fprime
from cardillo.math.rotations import (
    Exp_SO3_quat,
    Exp_SO3_quat_P,
    Log_SO3_quat,
    T_SO3_quat,
    T_SO3_quat_P,
    T_SO3_quat_Q_P,
    T_SO3_inv_quat,
    T_SO3_inv_quat_P,
    quatprod,
)
from cardillo.math.SmallestRotation import SmallestRotation
from cardillo.utility.coo_matrix import CooMatrix
from cardillo.utility.sparse_array_blocks import SparseArrayBlocks

from ._base_export import RodExportBase
from ._base_interface import RodInterface
from ._cross_section import CrossSectionInertias_new
from .discretization.lagrange import LagrangeKnotVector
from .discretization.mesh1D import Mesh1D_equidistant

zeros3 = np.zeros(3, dtype=float)
eye3 = np.eye(3, dtype=float)


####
# we do the following:
# N = nel + 1
# q0 = [r_OCi, P_IBi] in R^7
# qi = [r_OCi, P_IBi, q_eps] in R^8, i in [1, N-1]
# qN = [r_OCi, P_IBi] in R^7
# q = [q0, ...qN, alpha0, ... alpha(N-1)] in R^{7N + 2N - 3}
####


class KirchhoffLoveRod_PetrovGalerkin(RodInterface):
    def _create_meshs(self):
        mesh_rP = Mesh1D_equidistant(
            basis="Hermite_C0",
            nelement=self.nelement,
            polynomial_degere=3,
            derivative_order=2,
        )
        mesh_alpha = Mesh1D_equidistant(
            basis="Lagrange",
            nelement=self.nelement,
            polynomial_degere=2,
            derivative_order=1,
        )
        mesh_n = Mesh1D_equidistant(
            basis="Lagrange_Disc",
            nelement=self.nelement,
            polynomial_degere=2,
            derivative_order=0,
        )
        mesh_m = Mesh1D_equidistant(
            basis="Lagrange_Disc",
            nelement=self.nelement,
            polynomial_degere=1,
            derivative_order=0,
        )

        # element intervals
        self.element_interval = mesh_rP.element_interval
        self.element_number = mesh_rP.element_number
        self.node_number = mesh_rP.node_number

        # total number of nodes
        self.nnodes = self.nelement + 1
        self.nnodes_n = mesh_n.nnodes
        self.nnodes_m = mesh_m.nnodes

        self.h3 = lambda xis, els: mesh_rP.shape_functions(xis, els, 2)
        self.h3_element = lambda xi, el: mesh_rP.shape_function_array_element(xi, el, 0)

        self.N = lambda xis, els: mesh_alpha.shape_functions(xis, els, 1)
        self.N_element = lambda xi, el: mesh_alpha.shape_function_array_element(
            xi, el, 0
        )

        self.Nn = lambda xis, els: mesh_n.shape_functions(xis, els, 0)[0]
        self.Nm = lambda xis, els: mesh_m.shape_functions(xis, els, 0)[0]

        #####################
        # quadrature points #
        #####################
        # internal virtual work contribution
        quadrature_int_h = mesh_rP.quadrature(*self.quadrature_int, 2)
        self.nquadrature_int_total = quadrature_int_h["nquadrature_total"]
        self.qp_int_vec = quadrature_int_h["qp"]
        self.qw_int_vec = quadrature_int_h["qw"]
        self.qels_int_vec = quadrature_int_h["els"]
        self.h3_int, self.h3_xi_int, self.h3_xixi_int = quadrature_int_h["N"]

        quadrature_int_N = mesh_alpha.quadrature(*self.quadrature_int, 1)
        self.N_int, self.N_xi_int = quadrature_int_N["N"]

        quadrature_int_n = mesh_n.quadrature(*self.quadrature_int, 0)
        self.Nn_int = quadrature_int_n["N"][0]
        quadrature_int_m = mesh_m.quadrature(*self.quadrature_int, 0)
        self.Nm_int = quadrature_int_m["N"][0]

        self.shape_functions_int = (
            self.h3_int,
            self.h3_xi_int,
            self.h3_xixi_int,
            self.N_int,
            self.N_xi_int,
        )

        # inertial virtual work contributions
        quadrature_dyn_h = mesh_rP.quadrature(*self.quadrature_dyn, 2)
        self.nquadrature_dyn_total = quadrature_dyn_h["nquadrature_total"]
        self.qp_dyn_vec = quadrature_dyn_h["qp"]
        self.qw_dyn_vec = quadrature_dyn_h["qw"]
        self.qels_dyn_vec = quadrature_dyn_h["els"]
        self.h3_dyn, self.h3_xi_dyn, self.h3_xixi_dyn = quadrature_dyn_h["N"]

        quadrature_dyn_N = mesh_alpha.quadrature(*self.quadrature_dyn, 1)
        self.N_dyn, self.N_xi_dyn = quadrature_dyn_N["N"]

        self.shape_functions_dyn = (
            self.h3_dyn,
            self.h3_xi_dyn,
            self.h3_xixi_dyn,
            self.N_dyn,
            self.N_xi_dyn,
        )

        # external virtual work contributions
        quadrature_ext_h = mesh_rP.quadrature(*self.quadrature_ext, 2)
        self.nquadrature_ext_total = quadrature_ext_h["nquadrature_total"]
        self.qp_ext_vec = quadrature_ext_h["qp"]
        self.qw_ext_vec = quadrature_ext_h["qw"]
        self.qels_ext_vec = quadrature_ext_h["els"]
        self.h3_ext, self.h3_xi_ext, self.h3_xixi_ext = quadrature_ext_h["N"]

        quadrature_ext_N = mesh_alpha.quadrature(*self.quadrature_ext, 1)
        self.N_ext, self.N_xi_ext = quadrature_ext_N["N"]

        self.shape_functions_ext = (
            self.h3_ext,
            self.h3_xi_ext,
            self.h3_xixi_ext,
            self.N_ext,
            self.N_xi_ext,
        )

    def _post_init_(self):
        # M
        M_pairs = ...

        if self._nla_c > 0:
            # c_la_c
            if self.n_c:
                c_la_c_n_pairs = [
                    (self.Nn_int, self.Nn_int, self.qw_int_vec * self.J_int_vec)
                ]
                self.c_la_c_n_SAB = SparseArrayBlocks(
                    (self._nla_cn, self._nla_cn),
                    (1, 1),
                    c_la_c_n_pairs,
                )
            if len(self.idx_m_c) > 0:
                c_la_c_m_pairs = [
                    (self.Nm_int, self.Nm_int, self.qw_int_vec * self.J_int_vec)
                ]
                self.c_la_c_m_SAB = SparseArrayBlocks(
                    (self._nla_cm, self._nla_cm),
                    (len(self.idx_m_c), len(self.idx_m_c)),
                    c_la_c_m_pairs,
                )

        # c_sigma
        c_sigma_q_pairs = ...

        # W_sigma
        W_sigma_rt_n_pairs = [
            (self.h3_xi_int, self.Nn_int, self.qw_int_vec),
        ]
        self.W_sigma_rt_n_SAB = SparseArrayBlocks(
            (self.nu_rt, self.nnodes_n),
            (3, 1),
            W_sigma_rt_n_pairs,
        )
        W_sigma_rt_m_pairs = [
            (self.h3_xi_int, self.Nm_int, self.qw_int_vec),
            (self.h3_xixi_int, self.Nm_int, self.qw_int_vec),
        ]
        self.W_sigma_rt_m_SAB = SparseArrayBlocks(
            (self.nu_rt, 3 * self.nnodes_m),
            (3, 3),
            W_sigma_rt_m_pairs,
            [(..., self.idx_m_c), (..., self.idx_m_g)],
        )
        W_sigma_theta_m_pairs = [
            (self.N_int, self.Nm_int, self.qw_int_vec),
            (self.N_xi_int, self.Nm_int, self.qw_int_vec),
        ]
        self.W_sigma_theta_m_SAB = SparseArrayBlocks(
            (self.nu_theta, 3 * self.nnodes_m),
            (1, 3),
            W_sigma_theta_m_pairs,
            [(..., self.idx_m_c), (..., self.idx_m_g)],
        )

        # Wla_sigma_q
        h_pot_q_pairs = ...

    def _create_system_interfaces(self):
        # qDOF
        self.nq_r = 3 * self.nnodes
        self.nq_P = 4 * self.nnodes
        self.nq_eps = self.nnodes - 2
        self.nq_alpha = self.nelement
        nq = np.cumsum([self.nq_r, self.nq_P, self.nq_eps, self.nq_alpha])
        self.qDOF_r = slice(0, nq[0])
        self.qDOF_P = slice(nq[0], nq[1])
        self.qDOF_eps = slice(nq[1], nq[2])
        self.qDOF_alpha = slice(nq[2], nq[3])
        self.nq = nq[-1]

        # uDOF
        self.nu_r = 3 * self.nnodes
        self.nu_phi = 3 * self.nnodes
        self.nu_eps = 2 * self.nnodes - 2
        self.nu_alpha = self.nelement
        nu = np.cumsum([self.nu_r, self.nu_phi, self.nu_eps, self.nu_alpha])
        self.uDOF_r = slice(0, nu[0])
        self.uDOF_phi = slice(nu[0], nu[1])
        self.uDOF_eps = slice(nu[1], nu[2])
        self.uDOF_alpha = slice(nu[2], nu[3])
        self.nu = nu[-1]

        self.nu_rt = 3 * self.nnodes + 6 * self.nelement
        self.nu_theta = self.nnodes + self.nelement
        print(f"nu_rt: {self.nu_rt}")
        print(f"nu_theta: {self.nu_theta}")

        # total number of compliance coordinates
        self.nla_sigma = self.nnodes_n + self.nnodes_m * 3
        self._nla_cn = self.nnodes_n * self.n_c
        self._nla_gn = self.nnodes_n * self.n_g
        self._nla_cm = self.nnodes_m * len(self.idx_m_c)
        self._nla_gm = self.nnodes_m * len(self.idx_m_g)
        nla_c = self._nla_cn + self._nla_cm
        nla_g = self._nla_gn + self._nla_gm

        self._handle_internal(nla_c, nla_g)

    def set_parameter(
        self,
        *,
        cross_section=None,
        material_model=None,
        cross_section_inertias=None,
        distributed_load=None,
    ):
        if distributed_load is not None:
            assert (
                len(distributed_load) == 2
            ), "Line distributed forces must be a list of length 2 (force and moment)."
            assert distributed_load[1] is None, "No line distributed moment allowed."

        super().set_parameter(
            cross_section=cross_section,
            material_model=material_model,
            cross_section_inertias=cross_section_inertias,
            distributed_load=distributed_load,
        )

    @staticmethod
    def straight_configuration(
        nelement,
        L,
        r_OP0=zeros3,
        A_IB0=eye3,
    ):
        """Compute generalized position coordinates for straight configuration."""
        nnodes = nelement + 1

        x0 = np.linspace(0, L, num=nnodes)
        y0 = np.zeros(nnodes)
        z0 = np.zeros(nnodes)
        r_OP = np.vstack((x0, y0, z0))
        r = np.zeros((nnodes, 3), dtype=float)
        for i in range(nnodes):
            r[i] = r_OP0 + A_IB0 @ r_OP[:, i]

        p = np.tile(Log_SO3_quat(A_IB0) * np.sqrt(L), nnodes)
        eps = np.zeros(nelement - 1)
        alpha = np.zeros(nelement)
        return np.concat([r.reshape(-1), p.reshape(-1), eps, alpha])

    @staticmethod
    def pose_configuration(nelement, r_OP, A_IB, xi1=1, r_OP0=zeros3, A_IB0=eye3): ...

    @staticmethod
    def straight_initial_configuration(
        nelement, L, r_OP0=zeros3, A_IB0=eye3, v_P0=zeros3, B_omega_IB0=eye3
    ): ...

    ############################
    # export of centerline nodes
    ############################
    def nodes(self, qsystem):
        """Returns nodal position coordinates"""
        qbody = qsystem[self.qDOF]
        qnodesT = qbody[self.qDOF_r].reshape(-1, self.nnodes, order="F")
        return qnodesT[:3]

    def centerline(self, q, num=100):
        q_body = q[self.qDOF]
        xis, els = self._eval_logic(None, num)
        shape_functions = [*self.h3(xis, els), *self.N(xis, els)]
        r_OC, _ = self._eval_int_vec(shape_functions, q_body, choice="eval")
        return r_OC.T

    def frames(self, q, num=10):
        q_body = q[self.qDOF]
        xis, els = self._eval_logic(None, num)
        shape_functions = [*self.h3(xis, els), *self.N(xis, els)]
        r_OC, A_IB = self._eval_int_vec(shape_functions, q_body, choice="eval")

        return r_OC.T, A_IB[:, :, 0].T, A_IB[:, :, 1].T, A_IB[:, :, 2].T

    ##################
    # abstract methods
    ##################
    def assembler_callback(self):
        # self._M_coo()
        if self._nla_c > 0:
            self._c_la_c_coo()

    ############################################
    # interpolations and virtual work mappings #
    ############################################
    def compute_rt_P(self, q):
        # TODO: make this also work with qe
        # positions and tangents
        r = q[self.qDOF_r].reshape(self.nnodes, 3)
        P_IB = q[self.qDOF_P].reshape(self.nnodes, 4)
        eps = np.array([0, *q[self.qDOF_eps], 0])

        P2 = np.sum(P_IB**2, axis=1)
        A_IB = Exp_SO3_quat(P_IB)
        ex_B = A_IB[:, :, 0]

        scl_plus = (1 + eps) * P2
        scl_minus = (1 - eps) * P2

        t_plus = ex_B[:-1] * scl_plus[:-1, None]
        t_minus = ex_B[1:] * scl_minus[1:, None]

        rt = np.vstack([r, t_plus, t_minus])

        # quaternions
        P_unit = P_IB / np.sqrt(P2)[:, None]
        Px = np.empty((self.nelement, 4))
        Px[:, 0] = 1.0
        Px[:, 1] = q[self.qDOF_alpha]
        Px[:, 2:4] = 0.0

        P_mid = quatprod(0.5 * (P_unit[:-1] + P_unit[1:]), Px)
        P_ = np.zeros((self.nnodes + self.nelement, 4))
        P_[::2] = P_unit
        P_[1::2] = P_mid

        return rt, P_

    def _eval_int_vec(self, shape_functions, q, choice="eval", deval=False):
        rt, P = self.compute_rt_P(q)
        # centerline
        r_OC = shape_functions[0] @ rt
        r_OC_xi = shape_functions[1] @ rt

        j = np.sqrt(np.sum(r_OC_xi**2, axis=1))
        ex_B = r_OC_xi / j[:, None]

        # orientation A
        if self.orientation_type == "A":
            raise NotImplementedError
            rP = N @ qnodes
            r_CP = self._A_IB(rP[3:]) @ B_r_CP
            return rP[:3] + r_CP

            point_dict = self.get_interaction_point(xi)
            N = point_dict["N"]

            # TODO: difference in first and last element!
            qnodes = qi[self.DOF_nodes].reshape(2, -1)

            r_OC0 = qnodes[0, :3]
            r_OC1 = qnodes[1, :3]

            P_IB0 = qnodes[0, 3:7]
            P_IB1 = qnodes[1, 3:7]

            q_eps0 = qnodes[0, -1]
            q_eps1 = qnodes[1, -1]

            A_IB0 = Exp_SO3_quat(P_IB0)
            A_IB1 = Exp_SO3_quat(P_IB1)

            t0 = A_IB0[:, 2] * (1 + q_eps0)
            t1 = A_IB1[:, 2] * (1 - q_eps1)

            r_OC = h00 * r_OC0 + h01 * t0 + h10 * r_OC1 + h11 * t1

            if B_r_CP @ B_r_CP == 0.0:
                return r_OC

            # TODO: check if normalization is necessary
            P_IM = P_IB0 / np.linalg.norm(P_IB0) + P_IB1 / np.linalg.norm(P_IB1)
            A_IM = Exp_SO3_quat(P_IM)

            r_OC = h00 * r_OC0 + h01 * t0 + h10 * r_OC1 + h11 * t1
            r_OC_xi = h00_xi * r_OC0 + h01_xi * t0 + h10_xi * r_OC1 + h11_xi * t1

            A_MJ = SR(A_IM.T @ r_OC_xi, 0)

            alpha = N0 * SR_error(...) + N1 * qi[self.DOF_alpha] + N2 * SR_error(...)
            A_JB = A_IB_basic(alpha).x

            A_IB = A_IM @ A_MJ @ A_JB
            r_CP = A_IB @ B_r_CP
            # TODO: can we get A_IB instead by
            # P_IJm = P_IM oxo (np.sin(alpha/2), np.cos(alpha/2), 0, 0) or similar, maybe change DOF
            # then P_IJ = N0 * P_IB0 + N1 * P_IJm + N2 * P_IB1
            # then A_IJ = Exp_SO3(P_IJ)
            # and A_JB = SR(A_IJ.T @ r_OC_xi, 0)

            # TODO: compare complexity: where does curvature come from and how do pd's w.r.t. q look like?

            return r_OC + r_CP
            ...
        # TODO: implement this

        elif self.orientation_type == "B":
            # orientation B
            P_IJ = shape_functions[3] @ P
            A_IJ = Exp_SO3_quat(P_IJ)
            J_ex_B = np.einsum("ijk,ij->ik", A_IJ, ex_B)

            # TODO: vectorize SR
            SR_JB = [SmallestRotation(J_ex_Bi, 1) for J_ex_Bi in J_ex_B]
            A_JB = np.array([SR_JB.A_RJ for SR_JB in SR_JB])
            A_IB = A_IJ @ A_JB

        if choice == "eval":
            return r_OC, A_IB

        r_OC_xixi = shape_functions[2] @ rt
        j_xi = np.einsum("ij,ij->i", ex_B, r_OC_xixi)  # TODO: matmul or sum?
        ex_B_xi = (r_OC_xixi - np.einsum("ij,ik,ik->ij", ex_B, ex_B, r_OC_xixi)) / j[
            :, None
        ]
        ex_B_to_j = ex_B / j[:, None]
        ex_B_to_j_xi = ex_B_xi / j[:, None] - ex_B * (j_xi / j**2)[:, None]

        # if not deval:
        P_IJ_xi = shape_functions[4] @ P
        T_IJ = T_SO3_quat(P_IJ)
        J_kappa_bar_IJ = np.einsum("ijk,ik->ij", T_IJ, P_IJ_xi)
        dJ_ex_B = np.cross(J_ex_B, J_kappa_bar_IJ) + np.einsum(
            "ijk,ij->ik", A_IJ, ex_B_xi
        )
        T_JB = np.array([SR_JB.T_RJ for SR_JB in SR_JB])
        B_kappa_bar = np.einsum("ijk,ij->ik", A_JB, J_kappa_bar_IJ) + np.einsum(
            "ijk,ik->ij", T_JB, dJ_ex_B
        )

        if choice == "strains":
            return j, B_kappa_bar

        if choice == "int":
            return A_IB, j, B_kappa_bar, ex_B_xi, ex_B_to_j, ex_B_to_j_xi

        print("Not implemented yet")

    def _set_reference_strains(self, Q):
        self.Q = Q.copy()

        # internal virtual work contributions
        J, B_kappa0_bar = self._eval_int_vec(
            self.shape_functions_int, self.Q, choice="strains"
        )
        self.J_int_vec = J
        self.B_Gamma0_int = np.zeros((self.nquadrature_int_total, 3), dtype=float)
        self.B_Gamma0_int[:, 0] = 1.0  # J / J = 1.0
        self.B_kappa0_bar_int = B_kappa0_bar
        self.B_Kappa0_int = self.B_kappa0_bar_int / J[:, None]
        self.epsilon0_int = np.hstack([self.B_Gamma0_int, self.B_Kappa0_int])
        self._epsilon_int = np.zeros((self.nquadrature_int_total, 6))

        # inertial virtual work contributions
        J, _ = self._eval_int_vec(self.shape_functions_dyn, self.Q, choice="strains")
        self.J_dyn_vec = J

        # external virtual work contributions
        J, _ = self._eval_int_vec(self.shape_functions_ext, self.Q, choice="strains")
        self.J_ext_vec = J

    def _projection_D(self, q):
        # 3.60 but in different order
        D = np.zeros((2 * self.nnodes - 2, 3, 3))

        P_IB = q[self.qDOF_P].reshape(self.nnodes, 4)
        eps = np.array([0, *q[self.qDOF_eps], 0])

        P2 = np.sum(P_IB**2, axis=1)
        A_IB = Exp_SO3_quat(P_IB)

        scl_plus = (1 + eps) * P2
        scl_minus = (1 - eps) * P2

        mtx_plus = np.zeros((self.nnodes - 1, 3, 3))
        mtx_plus[:, 0, 0] = 1.0
        mtx_plus[:, 1, 2] = scl_plus[:-1]
        mtx_plus[:, 2, 1] = -scl_plus[:-1]

        mtx_minus = np.zeros((self.nnodes - 1, 3, 3))
        mtx_minus[:, 0, 0] = 1.0
        mtx_minus[:, 1, 2] = scl_minus[1:]
        mtx_minus[:, 2, 1] = -scl_minus[1:]

        # fist for t+, then for t-
        D[: self.nnodes - 1] = np.einsum("ijk,ikl->ijl", A_IB[:-1], mtx_plus)
        D[self.nnodes - 1 :] = np.einsum("ijk,ikl->ijl", A_IB[1:], mtx_minus)

        return D

    def _projection_D_matrix(self, q):
        # TODO: cache and optimize
        Di = self._projection_D(q)

        # P_IB = q[self.qDOF_P].reshape(self.nnodes, 4)
        # eps = np.array([0, *q[self.qDOF_eps], 0])

        # P2 = np.sum(P_IB**2, axis=1)
        # A_IB = Exp_SO3_quat(P_IB)

        # scl_plus = (1 + eps) * P2
        # scl_minus = (1 - eps) * P2

        block_cols = np.arange(self.nelement)
        indptr = np.arange(self.nelement + 1)
        shape = (3 * self.nelement, self.nelement)
        block_size = (3, 1)

        D_phi_y_plus = bsr_array(
            (Di[: self.nelement, :, 1, None], block_cols, indptr),
            shape=shape,
            blocksize=block_size,
        )
        D_phi_y_minus = bsr_array(
            (Di[self.nelement :, :, 1, None], block_cols, indptr),
            shape=shape,
            blocksize=block_size,
        )
        D_phi_z_plus = bsr_array(
            (Di[: self.nelement, :, 2, None], block_cols, indptr),
            shape=shape,
            blocksize=block_size,
        )
        D_phi_z_minus = bsr_array(
            (Di[self.nelement :, :, 2, None], block_cols, indptr),
            shape=shape,
            blocksize=block_size,
        )
        D_eps_plus = bsr_array(
            (Di[: self.nelement, :, 0, None], block_cols, indptr),
            shape=shape,
            blocksize=block_size,
        )
        D_eps_minus = bsr_array(
            (Di[self.nelement :, :, 0, None], block_cols, indptr),
            shape=shape,
            blocksize=block_size,
        )

        t_plus_row = slice(self.nu_r, self.nu_r + 3 * self.nelement)
        t_minus_row = slice(
            self.nu_r + 3 * self.nelement, self.nu_r + 6 * self.nelement
        )

        phi_y_plus_col = slice(self.uDOF_phi.start + 1, self.uDOF_phi.stop - 3, 3)
        phi_z_plus_col = slice(self.uDOF_phi.start + 2, self.uDOF_phi.stop - 3, 3)
        phi_y_minus_col = slice(self.uDOF_phi.start + 1 + 3, self.uDOF_phi.stop, 3)
        phi_z_minus_col = slice(self.uDOF_phi.start + 2 + 3, self.uDOF_phi.stop, 3)

        # TODO: how to order epsilons? (plus, minus) or (0, 1-, 1+, 2-, 2+, ...(N-1)-, (N-1)+, N)?
        # eps_plus_col = slice(self.uDOF_eps.start, self.uDOF_eps.stop, 2)
        # eps_minus_col = slice(self.uDOF_eps.start+1, self.uDOF_eps.stop, 2)
        eps_plus_col = slice(self.uDOF_eps.start, self.uDOF_eps.stop - self.nelement)
        eps_minus_col = slice(self.uDOF_eps.start + self.nelement, self.uDOF_eps.stop)

        D1 = CooMatrix((self.nu_rt, self.nu))
        D1["I", : self.nu_r, self.uDOF_r] = eye_array(self.nu_r)
        D1["phi_y+", t_plus_row, phi_y_plus_col] = D_phi_y_plus
        D1["phi_y-", t_minus_row, phi_y_minus_col] = D_phi_y_minus
        D1["phi_z+", t_plus_row, phi_z_plus_col] = D_phi_z_plus
        D1["phi_z-", t_minus_row, phi_z_minus_col] = D_phi_z_minus
        D1["eps+", t_plus_row, eps_plus_col] = D_eps_plus
        D1["eps-", t_minus_row, eps_minus_col] = D_eps_minus

        # TODO: precompute
        phi_x_col = slice(self.uDOF_phi.start, self.uDOF_phi.stop, 3)
        phi_x_row = slice(0, self.nu_theta, 2)
        alpha_row = slice(1, self.nu_theta - 1, 2)
        D2 = CooMatrix((self.nu_theta, self.nu))
        D2["phi_x", phi_x_row, phi_x_col] = eye_array(self.nnodes)
        D2["alpha", alpha_row, self.uDOF_alpha] = eye_array(self.nelement)

        return D1, D2

    def process_h_node(self, q, h_rt, h_theta):
        # forces on (delta j_pm, delta phi_y, delta phi_z)
        D = self._projection_D(q)
        h_rt[self.nnodes :] = np.einsum(
            "ikj,ik->ij", D, h_rt[self.nnodes :]
        )  # D-transpose

        h_j_pm = h_rt[self.nnodes :, 0]
        h_psi = np.zeros((self.nnodes, 3))
        h_psi[:, 0] = h_theta[::2]
        h_psi[:-1, 1:] = h_rt[self.nnodes : 2 * self.nnodes - 1, 1:]
        h_psi[1:, 1:] += h_rt[2 * self.nnodes - 1 :, 1:]

        h_alpha = h_theta[1::2]
        return np.concatenate(
            [h_rt[: self.nnodes].reshape(-1), h_psi.reshape(-1), h_j_pm, h_alpha]
        )

    ############################
    # total energies and momenta
    ############################
    def _E_pot_comp(self, t, q, la_c):
        warn("Not implemented yet")
        return 0.0

    def E_pot_int(self, t, q):
        warn("Not implemented yet")
        return 0.0

    def E_kin(self, t, q, u):
        warn("Not implemented yet")
        return 0.0

    def linear_momentum(self, t, q, u):
        warn("Not implemented yet")
        return zeros3

    def angular_momentum(self, t, q, u):
        warn("Not implemented yet")
        return zeros3

    #########################################
    # equations of motion
    #########################################
    def M(self, t, q):
        warn("Not implemented yet")
        return np.eye(self.nu)

    def f_gyr(self, t, q, u):
        warn("Not implemented yet")
        return np.zeros(self.nu)

    def f_gyr_u(self, t, q, u):
        warn("Not implemented yet")
        return np.zeros((self.nu, self.nu))

    ####################################################
    # interactions with other bodies and the environment
    ####################################################
    def get_interaction_point(self, xi):
        # TODO: call this from constraints, see Tianxiang Marker, so that it is not called from postprocessing r_OP, etc. calls
        # TODO: check that it is always done using this function, never access interaction points directly!
        if not (xi in self.interaction_points.keys()):
            if (node_number := self.node_number(xi)) is not False:
                qDOF_r = uDOF_r = np.arange(3) + 3 * node_number
                qDOF_P = np.arange(4) + 4 * node_number + 3 * self.nnodes
                uDOF_phi = np.arange(3) + 3 * node_number + 3 * self.nnodes
                qDOF = np.concatenate([qDOF_r, qDOF_P])
                uDOF = np.concatenate([uDOF_r, uDOF_phi])

                Nq = np.eye(7)
                Nu = np.eye(6)

            else:
                raise NotImplementedError("xi must be at a node!")

            self.interaction_points[xi] = dict(
                qDOF=qDOF,
                uDOF=uDOF,
                r_q=Nq[:3],
                P_q=Nq[3:],
                J_C=Nu[:3],
                B_J_R=Nu[3:],
                zero_3_nui_nqi=np.zeros((3, 6, 7), dtype=float),
            )
        return self.interaction_points[xi]

    ##########################
    # r_OP / A_IB contribution
    ##########################
    def r_OP(self, t, qi, xi, B_r_CP=zeros3):
        point_dict = self.get_interaction_point(xi)
        r_OC = qi[:3]
        if B_r_CP @ B_r_CP == 0.0:
            return r_OC

        P_IB = qi[3:]
        return r_OC + Exp_SO3_quat(P_IB) @ B_r_CP

    def r_OP_q(self, t, qi, xi, B_r_CP=zeros3):
        # return approx_fprime(qi, lambda qi_: self.r_OP(t, qi_, xi, B_r_CP))
        point_dict = self.get_interaction_point(xi)
        if B_r_CP @ B_r_CP == 0.0:
            return point_dict["r_q"]

        r_CP_q = np.einsum("ijk,j->ik", self.A_IB_q(t, qi, xi), B_r_CP)
        return point_dict["r_q"] + r_CP_q

    def J_P(self, t, qi, xi, B_r_CP=zeros3):
        point_dict = self.get_interaction_point(xi)
        if B_r_CP @ B_r_CP == 0.0:
            return point_dict["J_C"]

        P = qi[3:]
        B_J_CP = np.cross(
            -B_r_CP[:, None], point_dict["B_J_R"], axisa=0, axisb=0, axisc=0
        )
        return point_dict["J_C"] + Exp_SO3_quat(P) @ B_J_CP

    def J_P_q(self, t, qi, xi, B_r_CP=zeros3):
        # return approx_fprime(qi, lambda qi_: self.J_P(t, qi_, xi, B_r_CP))
        point_dict = self.get_interaction_point(xi)
        if B_r_CP @ B_r_CP == 0.0:
            return point_dict["zero_3_nui_nqi"]

        B_J_CP = np.cross(
            -B_r_CP[:, None], point_dict["B_J_R"], axisa=0, axisb=0, axisc=0
        )
        return np.einsum("ijk, jl -> ilk", self.A_IB_q(t, qi, xi), B_J_CP)

    def A_IB(self, t, qi, xi):
        point_dict = self.get_interaction_point(xi)
        P_IB = qi[3:]
        return Exp_SO3_quat(P_IB)

    def A_IB_q(self, t, qi, xi):
        # return approx_fprime(qi, lambda qi_: self.A_IB(t, qi_, xi))
        point_dict = self.get_interaction_point(xi)
        P_IB = qi[3:]
        return Exp_SO3_quat_P(P_IB) @ point_dict["P_q"]

    def B_J_R(self, t, qi, xi):
        point_dict = self.get_interaction_point(xi)
        return point_dict["B_J_R"]

    def B_J_R_q(self, t, qi, xi):
        # return approx_fprime(qi, lambda qi_: self.B_J_R(t, qi_, xi))
        point_dict = self.get_interaction_point(xi)
        return point_dict["zero_3_nui_nqi"]

    #########################
    # external virtual work #
    # by distributed load   #
    #########################
    def E_pot_ext(self, t, q):
        r_OC, _ = self._eval_int_vec(self.shape_functions_ext, q, choice="eval")
        b_qp = self.distributed_load[0](t, self.qp_ext_vec)
        return -np.einsum("ij,ij", r_OC, b_qp * self.weights_ext[:, None])

    def f_ext(self, t, q, u):
        b_qp = self.distributed_load[0](t, self.qp_ext_vec)
        h_rt = self.N_ext.T @ (b_qp * self.weights_ext[:, None])
        return self.process_h_node(
            q, h_rt, np.zeros(2 * self.nelement + 1, dtype=float)
        )

    #########################
    # internal virtual work #
    #########################
    def l_sigma(self, q):
        j, B_kappa_bar = self._eval_int_vec(
            self.shape_functions_int, q[self.qDOF], choice="strains"
        )

        ln = self.Nn.T @ ((j - self.J_int_vec) * self.qw_int_vec[:, None])
        lm = self.Nm.T @ (
            (B_kappa_bar - self.B_kappa0_bar_int) * self.qw_int_vec[:, None]
        )

        lm_c = lm[:, self.idx_m_c].reshape(-1)
        lm_g = lm[:, self.idx_m_g].reshape(-1)
        l_c = np.concatenate([ln.reshape(-1), lm_c]) if self.n_c else lm_c
        l_g = np.concatenate([ln.reshape(-1), lm_g]) if self.n_g else lm_g
        return l_c, l_g

    def l_sigma_q(self, q):
        l_c_q = approx_fprime(q, lambda q_: self.l_sigma(q_)[0])
        l_g_q = approx_fprime(q, lambda q_: self.l_sigma(q_)[1])
        return l_c_q, l_g_q

    def _c_la_c_coo(self):
        C_qp = self.material_model.C_inv(self.qp_int_vec, quadrature=True)

        if self.n_c and len(self.idx_m_c) > 0:
            C_coupled = np.zeros_like(C_qp)
            C_coupled[:, 0, 3:] = C_qp[:, 0, 3:]
            C_coupled[:, 3:, 0] = C_qp[:, 3:, 0]

            error = np.linalg.norm(C_coupled)
            if error != 0.0:
                warn("There are couple terms in C, which are neglected!")

        c_la_c = CooMatrix((self.nla_c, self.nla_c))
        if self.n_c:
            C_n = C_qp[:, 0, 0]
            c_la_c[: self._nla_cn, : self._nla_cn] = self.c_la_c_n_SAB.add_blocks(
                C_n[None, :, None, None]
            )

        if len(self.idx_m_c) > 0:
            C_m = C_qp[:, 3:, 3:]
            c_la_c[self._nla_cn :, self._nla_cn :] = self.c_la_c_m_SAB.add_blocks(
                C_m[None, :, self.idx_m_c[:, None], self.idx_m_c]
            )

        self._cla_c = c_la_c.tocsc()
        self._cla_c_inv = spsolve(self._cla_c, eye_array(self.nla_c, format="csc"))

        return c_la_c

    def W_sigma(self, q):
        # compute W_sigma
        A_IB, j, B_kappa_bar, ex_B_xi, ex_B_to_j, ex_B_to_j_xi = self._eval_int_vec(
            self.shape_functions_int, q, choice="int"
        )

        # on rt -> r, j, phi
        W_rt_n_qp = A_IB[:, :, 0]
        W_rt_m_qp = np.empty((2, self.nquadrature_int_total, 3, 3))
        W_rt_m_qp[0] = -np.cross(
            ex_B_to_j_xi[:, None, :], A_IB
        )  # TODO: avoid "-"-sign?
        W_rt_m_qp[1] = -np.cross(ex_B_to_j[:, None, :], A_IB)  # TODO: avoid "-"-sign?

        # TODO: I think it is better to keep c and g together and split up later the Coo
        W_sigma_rt_n = self.W_sigma_rt_n_SAB.add_blocks(W_rt_n_qp[None, :, :, None])
        W_sigma_rt_m = self.W_sigma_rt_m_SAB.add_blocks(W_rt_m_qp)

        W_theta_m_qp = np.zeros((2, self.nquadrature_int_total, 3))
        W_theta_m_qp[0] = np.einsum("ij,ijk->ik", ex_B_xi, A_IB)
        W_theta_m_qp[1, 0] = 1.0
        W_sigma_theta_m = self.W_sigma_theta_m_SAB.add_blocks(
            W_theta_m_qp[:, :, None, :]
        )

        # TODO: prebuild this matrix
        W_sigma = CooMatrix((self.nla_sigma, self.nla_sigma))
        D1, D2 = self._projection_D_matrix(q)
        W_sigma["rt_n", : -self.nu_alpha, : self.nnodes_n] = D1.T @ W_sigma_rt_n
        W_sigma["rt_m", : -self.nu_alpha, self.nnodes_n :] = D1.T @ W_sigma_rt_m

        W_sigma["theta_m", :, self.nnodes_n] = D2.T @ W_sigma_theta_m

        return W_sigma

    def Wla_sigma(self, q, la_c=None, la_g=None): ...
    def h_pot(self, _eval, sigma_qp): ...

    def f_pot(self, t, q, u):
        A_IB, j, B_kappa_bar, ex_B_xi, ex_B_to_j, ex_B_to_j_xi = self._eval_int_vec(
            self.shape_functions_int, q, choice="int"
        )

        self._epsilon_int[:, 0] = j / self.J_int_vec
        self._epsilon_int[:, 3:] = B_kappa_bar / self.J_int_vec[:, None]

        sigma = self.material_model.sigma(
            self.qp_int_vec, self._epsilon_int, self.epsilon0_int, quadrature=True
        )
        n = sigma[:, 0]
        B_m = sigma[:, 3:]
        m = np.einsum("ijk,ik->ij", A_IB, B_m)

        ex_B = A_IB[:, :, 0]

        F0 = ex_B * n[:, None] - np.cross(ex_B_to_j_xi, m)
        F1 = -np.cross(ex_B_to_j, m)

        F2 = np.einsum("ij,ij->i", ex_B_xi, m)
        F3 = B_m[:, 0]

        # forces on I_detla r and I_delta_t_pm
        # order (r, t_plus, t_minus)
        h_rt = self.h3_xi_int.T @ (
            F0 * self.qw_int_vec[:, None]
        ) + self.h3_xixi_int.T @ (F1 * self.qw_int_vec[:, None])
        # moments on delta phi
        h_theta = self.N_int.T @ (F2 * self.qw_int_vec) + self.N_xi_int.T @ (
            F3 * self.qw_int_vec
        )

        return self.process_h_node(q, -h_rt, -h_theta)

    def Wla_sigma_q(self, q, la_c=None, la_g=None): ...

    def f_pot_q(self, t, q, u):
        return approx_fprime(q, lambda q_: self.f_pot(t, q_, u))

    ########################
    # evaluation functions #
    ########################
    def _eval_logic(self, n_per_element, n_ges):
        assert (n_per_element is not None) != (
            n_ges is not None
        ), "Either n_per_element or n_ges must be specified (not both)"

        if n_ges is not None:
            xis = np.linspace(0, 1, n_ges)
            els = self.element_number(xis)

        else:
            xis = []
            els = []
            for el in range(self.nelement):
                element_interval = self.element_interval[el]
                xis.append(
                    np.linspace(element_interval[0], element_interval[1], n_per_element)
                )
                els.append(np.tile(el, n_per_element))
            xis = np.concatenate(xis)
            els = np.concatenate(els)

        return xis, els

    def eval_stresses(self, *args, **kwargs):
        warn("Eval stresses returns strains!")
        xis, Ga, Ka = self.eval_strains(*args, **kwargs)
        Ga[:, 1:] = np.nan
        return xis, Ga, Ka

    def eval_strains(self, t, q, la_c, la_g, n_per_element=None, n_ges=None):
        xis, els = self._eval_logic(n_per_element, n_ges)

        epsilon = np.zeros((len(xis), 6))
        if self._nDB > 0:
            shape_functions = [*self.h3(xis, els), *self.N(xis, els)]
            # reference strains
            J, B_kappa0_bar = self._eval_int_vec(
                shape_functions, self.Q, choice="strains"
            )
            # current strains
            j, B_kappa_bar = self._eval_int_vec(
                shape_functions, q[self.qDOF], choice="strains"
            )

            epsilon[:, 0] = (j - J) / J
            epsilon[:, 3:] = (B_kappa_bar - B_kappa0_bar) / J[:, None]

        # # strains from compliance
        # if self._nla_c > 0:
        #     Nc = self.Nc(xis, els)
        #     la_c_nodes = la_c[self.la_cDOF].reshape(self.nnodes_sigma, -1)
        #     la_sigma = Nc @ la_c_nodes

        #     C_inv = self.material_model.C_inv[self.idx_c[:, None], self.idx_c]
        #     # TODO: position dependent material law
        #     epsilon[:, self.idx_c] = la_sigma @ C_inv.T

        # # strains from constraints are always 0
        # epsilon[:, self.idx_g] = 0.0

        return xis, epsilon[:, :3], epsilon[:, 3:]


def make_KirchhoffLoveRod(
    *,
    idx_constraints=None,
    idx_displacement_based=None,
    quadrature_int=None,
    quadrature_dyn=None,
    quadrature_ext=None,
    full_inertia=False,
    rotation_interpolation=None,
):
    """idx_constraints and idx_displacement_based in [0, 1, 2, 3]"""
    # constraints
    if idx_constraints is not None:
        idx_constraints = np.asarray(idx_constraints, dtype=int)
        if not ((idx_constraints >= 0).all() & (idx_constraints <= 3).all()):
            raise ValueError("constraint values must between 0 and 3")
    else:
        idx_constraints = np.array([], dtype=int)
    idx_constraints = np.sort(idx_constraints)

    # displacement based
    if idx_displacement_based is not None:
        idx_displacement_based = np.asarray(idx_displacement_based, dtype=int)
        if not (
            (idx_displacement_based >= 0).all() & (idx_displacement_based <= 3).all()
        ):
            raise ValueError("displacement_based values must between 0 and 3")
    else:
        idx_displacement_based = np.array([], dtype=int)
    idx_displacement_based = np.sort(idx_displacement_based)

    # check that no index is in both lists
    inter_g_DB = np.intersect1d(idx_constraints, idx_displacement_based)
    assert (
        inter_g_DB.size == 0
    ), f"the index {inter_g_DB} is both constrained and displacement based"

    # quadrature
    if quadrature_int == None:
        quadrature_int = (3, "Gauss")
    elif isinstance(quadrature_int, int):
        quadrature_int = (quadrature_int, "Gauss")
    elif not isinstance(quadrature_int, tuple):
        raise ValueError(
            "quadrature_int must be either an 'None', an integer for Gauss quadrature or a tuple: (nquadrature, method)"
        )

    if quadrature_dyn == None:
        # TODO: also look on what to do with full inertia
        n_full = 5
        quadrature_dyn = (n_full, "Gauss")
        print(f"quadrature_dyn: {quadrature_dyn}")
    elif isinstance(quadrature_dyn, int):
        quadrature_dyn = (quadrature_dyn, "Gauss")
    elif not isinstance(quadrature_dyn, tuple):
        raise ValueError(
            "quadrature_dyn must be either an 'None', an integer for Gauss quadrature or a tuple: (nquadrature, method)"
        )

    quadrature_ext = quadrature_dyn
    # if quadrature_ext == None:
    #     # TODO: take trapezoidal rule as default?
    #     quadrature_ext = (polynomial_degree + 1, "Trapezoidal")
    #     n_full = int(np.ceil(3 / 2 * polynomial_degree + 1 / 2))
    #     quadrature_ext = (n_full, "Gauss")
    #     print(f"quadrature_ext: {quadrature_ext}")
    # elif isinstance(quadrature_ext, int):
    #     quadrature_ext = (quadrature_ext, "Gauss")
    # elif not isinstance(quadrature_ext, tuple):
    #     raise ValueError(
    #         "quadrature_ext must be either an 'None', an integer for Gauss quadrature or a tuple: (nquadrature, method)"
    #     )

    # parametrization
    if rotation_interpolation is None:
        rotation_interpolation = "B"
    assert rotation_interpolation in ["A", "B"]

    class KirchhoffLoveRod(KirchhoffLoveRod_PetrovGalerkin):
        def _pre_init_(self):
            self.orientation_type = rotation_interpolation

            self.idx_g = idx_constraints
            self.idx_db = idx_displacement_based
            self.idx_c = np.setdiff1d(
                np.arange(4), np.union1d(idx_constraints, idx_displacement_based)
            )

            self.n_c = 0 in self.idx_c
            self.n_g = 0 in self.idx_g
            self.idx_m_c = np.array([i for i in [0, 1, 2] if i + 1 in self.idx_c])
            self.idx_m_g = np.array([i for i in [0, 1, 2] if i + 1 in self.idx_g])

            self.quadrature_int = quadrature_int
            self.quadrature_dyn = quadrature_dyn
            self.quadrature_ext = quadrature_ext

    return KirchhoffLoveRod
