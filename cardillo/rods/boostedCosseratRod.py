import numpy as np
from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey
import sparse
from scipy.sparse import (
    block_diag,
    bsr_array,
    csr_array,
    eye_array,
)
from scipy.sparse.linalg import spsolve
from warnings import warn

from cardillo.math.algebra import norm, cross3, ax2skew
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

from ._base_interface import RodInterface
from ._base_export import RodExportBase
from ._cross_section import CrossSectionInertias_new
from .discretization.mesh1D import Mesh1D_equidistant

zeros3 = np.zeros(3, dtype=float)
eye3 = np.eye(3, dtype=float)

class CosseratRod_PetrovGalerkin(RodInterface):
    def _create_meshs(self):
        mesh_kin = Mesh1D_equidistant(
            basis="Lagrange",
            nelement=self.nelement,
            polynomial_degere=self.polynomial_degree,
            derivative_order=1,
        )
        mesh_cg = Mesh1D_equidistant(
            basis="Lagrange_Disc",
            nelement=self.nelement,
            polynomial_degere=self.polynomial_degree - 1,
            derivative_order=0,
        )

        # element intervals
        self.element_interval = mesh_kin.element_interval
        self.element_number = mesh_kin.element_number
        self.node_number = mesh_kin.node_number

        # total number of nodes and per element
        self.nnodes = mesh_kin.nnodes
        self.nnodes_sigma = mesh_cg.nnodes

        self.N = lambda xis, els: mesh_kin.shape_functions(xis, els, 1)
        self.N_element = lambda xi, el: mesh_kin.shape_function_array_element(xi, el, 0)
        self.Nc = lambda xis, els: mesh_cg.shape_functions(xis, els, 0)[0]

        #####################
        # quadrature points #
        #####################
        # internal virtual work contributions
        quadrature_int_kin = mesh_kin.quadrature(*self.quadrature_int, 1)
        self.nquadrature_int_total = quadrature_int_kin["nquadrature_total"]
        self.qp_int_vec = quadrature_int_kin["qp"]
        self.qw_int_vec = quadrature_int_kin["qw"]
        self.qels_int_vec = quadrature_int_kin["els"]
        self.N_int, self.N_xi_int = quadrature_int_kin["N"]

        quadrature_int_cg = mesh_cg.quadrature(*self.quadrature_int, 0)
        self.Nc_int = quadrature_int_cg["N"][0]

        # inertial virtual work contributions
        quadrature_dyn = mesh_kin.quadrature(*self.quadrature_dyn, 1)
        self.nquadrature_dyn_total = quadrature_dyn["nquadrature_total"]
        self.qp_dyn_vec = quadrature_dyn["qp"]
        self.qw_dyn_vec = quadrature_dyn["qw"]
        self.qels_dyn_vec = quadrature_dyn["els"]
        self.N_dyn, self.N_xi_dyn = quadrature_dyn["N"]

        # external virtual work contributions
        quadrature_ext = mesh_kin.quadrature(*self.quadrature_ext, 1)
        self.nquadrature_ext_total = quadrature_ext["nquadrature_total"]
        self.qp_ext_vec = quadrature_ext["qp"]
        self.qw_ext_vec = quadrature_ext["qw"]
        self.qels_ext_vec = quadrature_ext["els"]
        self.N_ext, self.N_xi_ext = quadrature_ext["N"]

    def _set_reference_strains(self, Q):
        self.Q = Q.copy()

        # internal virtual work contributions
        _, B_gamma0_bar, B_kappa0_bar = self._eval_internal_vec(
            self.N_int, self.N_xi_int, self.Q
        )
        self.J_int_vec = np.linalg.norm(B_gamma0_bar, axis=1)
        self.B_gamma0_bar_int = B_gamma0_bar
        self.B_kappa0_bar_int = B_kappa0_bar
        self.epsilon0_int = (
            np.hstack([B_gamma0_bar, B_kappa0_bar]) / self.J_int_vec[:, None]
        )

        # inertial virtual work contributions
        _, B_gamma_bar, _ = self._eval_internal_vec(self.N_dyn, self.N_xi_dyn, self.Q)
        self.J_dyn_vec = np.linalg.norm(B_gamma_bar, axis=1)

        # external virtual work contributions
        _, B_gamma_bar, _ = self._eval_internal_vec(self.N_ext, self.N_xi_ext, self.Q)
        J_ext_vec = np.linalg.norm(B_gamma_bar, axis=1)
        self.weights_ext = J_ext_vec * self.qw_ext_vec

    def _post_init_(self):
        # M
        M_pairs = [(self.N_dyn, self.N_dyn, self.qw_dyn_vec * self.J_dyn_vec)]
        self.M_h_u_SAB = SparseArrayBlocks((self.nu, self.nu), (6, 6), M_pairs)

        if self._nla_c > 0:
            # c_la_c
            c_la_c_pairs = [
                (self.Nc_int, self.Nc_int, self.qw_int_vec * self.J_int_vec)
            ]
            self.c_la_c_SAB = SparseArrayBlocks(
                (self.nla_c, self.nla_c),
                (len(self.idx_c), len(self.idx_c)),
                c_la_c_pairs,
            )

        # c_sigma
        c_sigma_q_pairs = [
            (self.Nc_int, self.N_int, self.qw_int_vec),
            (self.Nc_int, self.N_xi_int, self.qw_int_vec),
        ]
        self.c_sigma_q_SAB = SparseArrayBlocks(
            (self.nla_sigma, self.nq),
            (6, self.nq_node),
            c_sigma_q_pairs,
            [(self.idx_c, ...), (self.idx_g, ...)],
        )

        # W_sigma
        W_sigma_pairs = [
            (self.N_int, self.Nc_int, self.qw_int_vec),
            (self.N_xi_int, self.Nc_int, self.qw_int_vec),
        ]
        self.W_sigma_SAB = SparseArrayBlocks(
            (self.nu, self.nla_sigma),
            (6, 6),
            W_sigma_pairs,
            [(..., self.idx_c), (..., self.idx_g)],
        )

        # Wla_sigma_q
        h_pot_q_pairs = [
            (self.N_int, self.N_int, self.qw_int_vec),
            (self.N_int, self.N_xi_int, self.qw_int_vec),
            (self.N_xi_int, self.N_int, self.qw_int_vec),
            (self.N_xi_int, self.N_xi_int, self.qw_int_vec),
        ]
        self.h_pot_q_SAB = SparseArrayBlocks(
            (self.nu, self.nq), (6, self.nq_node), h_pot_q_pairs
        )

    def _create_system_interfaces(self):
        # total number of generalized position and velocity coordinates
        self.nq = self.nnodes * self.nq_node
        self.nu = self.nnodes * 6

        # total number of compliance coordinates
        self.nla_sigma = self.nnodes_sigma * 6
        nla_c = self.nnodes_sigma * len(self.idx_c)
        nla_g = self.nnodes_sigma * len(self.idx_g)
        self._handle_internal(nla_c, nla_g)

        # g_S constraints
        self.nla_S = self.nnodes * self.dim_g_S
        self.la_S0 = np.zeros(self.nla_S, dtype=float)
        if self.nq_node == 12:
            warn("g_S rows and columns are completely wrong!")
        self._g_S_q_row = np.repeat(np.arange(self.nnodes), 4)
        self._g_S_q_col = (
            (3 + 7 * np.arange(self.nnodes))[:, None] + np.arange(4)
        ).ravel()

    def _eval_internal_vec(self, N, N_xi, q, deval=False):
        qbar_nodes = q.reshape(self.nnodes, -1)
        P_IB = N @ qbar_nodes[:, 3:]
        qbar_xi = N_xi @ qbar_nodes

        A_IB = self._A_IB(P_IB)
        T = self._T_IB(P_IB)

        B_gamma_bar = np.einsum("ijk,ij->ik", A_IB, qbar_xi[:, :3])
        B_kappa_bar = np.einsum("ijk,ik->ij", T, qbar_xi[:, 3:])
        if not deval:
            return A_IB, B_gamma_bar, B_kappa_bar

        # using my magic property
        B_gamma_bar_P = np.cross(B_gamma_bar[:, :, None], T, axisa=1, axisb=1, axisc=1)

        T_IB_P = self._T_IB_P(P_IB)
        B_kappa_bar_P = np.einsum("ijkl,ik->ijl", T_IB_P, qbar_xi[:, 3:])
        return (A_IB, B_gamma_bar, B_kappa_bar), (T, B_gamma_bar_P, B_kappa_bar_P)

    ############################
    # export of centerline nodes
    ############################
    def nodes(self, qsystem):
        """Returns nodal position coordinates"""
        qbody = qsystem[self.qDOF]
        qnodesT = qbody.reshape(-1, self.nnodes, order="F")
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
        els = self.element_number(xis)
        N = self.N(xis, els)[0]
        q_body = q[self.qDOF]
        q_nodes = q_body.reshape(self.nnodes, -1)
        r_OC = N @ q_nodes[:, :3]
        return r_OC.T

    def frames(self, q, num=10):
        xis = np.linspace(0, 1, num)
        els = self.element_number(xis)
        N = self.N(xis, els)[0]
        q_body = q[self.qDOF]
        q_nodes = q_body.reshape(self.nnodes, -1)
        rP = N @ q_nodes
        A_IB = self._A_IB(rP[:, 3:])
        return rP[:, :3].T, A_IB[:, :, 0].T, A_IB[:, :, 1].T, A_IB[:, :, 2].T

    ##################
    # abstract methods
    ##################
    def assembler_callback(self):
        self._M_coo()
        if self._nla_c > 0:
            self._c_la_c_coo()

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        qnodes = q.reshape(self.nnodes, self.nq_node)
        unodes = u.reshape(self.nnodes, 6)

        qnodes_dot = np.empty((self.nnodes, self.nq_node), dtype=np.common_type(q, u))
        qnodes_dot[:, :3] = unodes[:, :3]
        qnodes_dot[:, 3:] = np.einsum(
            "ijk,ik->ij", self._T_IB_inv(qnodes[:, 3:]), unodes[:, 3:]
        )

        return qnodes_dot.reshape(-1)

    def q_dot_q(self, t, q, u):
        # qnodes = q.reshape(self.nnodes, self.nq_node)
        unodes = u.reshape(self.nnodes, 6)

        blocks = np.empty((self.nnodes, self.nq_node, self.nq_node))
        blocks[:, :3] = 0.0
        blocks[:, 3:, :3] = 0.0
        blocks[:, 3:, 3:] = np.einsum(
            "jkl,ik->ijl",
            self._T_IB_inv_P,
            unodes[:, 3:],
        )

        return csr_array(block_diag(blocks))

    def q_dot_u(self, t, q):
        qnodes = q.reshape(self.nnodes, self.nq_node)

        blocks = np.empty((self.nnodes, self.nq_node, 6))
        blocks[:, :3, :3] = eye3
        blocks[:, :3, 3:] = 0.0
        blocks[:, 3:, :3] = 0.0
        blocks[:, 3:, 3:] = self._T_IB_inv(qnodes[:, 3:])

        return csr_array(block_diag(blocks))  # this keeps the 0's

    ############################
    # total energies and momenta
    ############################
    # the potential energies only work if
    # A) there is no coupling between the deformations
    # B) all of the coupled deformations are (not) db
    #    --> we put a warning when there are db and mx deformations
    def _E_pot_comp(self, t, q, la_c):
        if self._nDB > 0:
            msg = "E_pot_comp might not be correct if there are displacement-based deformations."
            warn(msg)
        _eval = self._eval_internal_vec(self.N_int, self.N_xi_int, q)
        epsilon_db = np.hstack([_eval[1], _eval[2]]) / self.J_int_vec[:, None]
        d_epsilon = epsilon_db - self.epsilon0_int

        la_sigma_nodes = np.zeros((self.nnodes_sigma, 6))
        la_sigma_nodes[:, self.idx_c] = la_c.reshape(self.nnodes_sigma, -1)
        la_sigma = self.Nc_int @ la_sigma_nodes

        C_qp = self.material_model.C_inv(self.qp_int_vec, quadrature=True)
        E_pot_i_star = 0.5 * np.einsum("ij,ijk,ik->i", la_sigma, C_qp, la_sigma)
        E_pot_i = np.sum(d_epsilon * la_sigma, axis=1) - E_pot_i_star
        E_pot = np.sum(E_pot_i * self.qw_int_vec * self.J_int_vec)
        return E_pot

    def E_pot_int(self, t, q):
        if self._nla_c > 0:
            warn("E_pot might not be correct if there are compliant deformations.")
        if self._nla_g > 0:
            warn("E_pot might not be correct if there are constrained deformations.")
        _eval = self._eval_internal_vec(self.N_int, self.N_xi_int, q)
        epsilon_db = np.hstack([_eval[1], _eval[2]]) / self.J_int_vec[:, None]
        epsilon = np.zeros_like(epsilon_db)
        epsilon[:, self.idx_db] = epsilon_db[:, self.idx_db]
        epsilon0 = np.zeros_like(epsilon_db)
        epsilon0[:, self.idx_db] = self.epsilon0_int[:, self.idx_db]
        E_pot_i = self.material_model.potential(
            self.qp_int_vec, epsilon, epsilon0, quadrature=True
        )
        E_pot = np.sum(E_pot_i * self.qw_int_vec * self.J_int_vec)
        return E_pot

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

    def linear_momentum(self, t, q, u):
        unodes = u.reshape(self.nnodes, -1)
        v = self.N_dyn @ unodes[:, :3]
        linear_momentum = np.sum(
            v * (self.A_rho0_qp * self.J_dyn_vec * self.qw_dyn_vec)[:, None], axis=0
        )
        return linear_momentum

    def angular_momentum(self, t, q, u):
        qnodes = q.reshape(self.nnodes, -1)
        rP = self.N_dyn @ qnodes
        r_OC = rP[:, :3]
        A_IB = self._A_IB(rP[:, 3:])

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

    #########################################
    # equations of motion
    #########################################
    def M(self, t, q):
        return self.__M

    def _M_coo(self):
        self.constant_mass_matrix = True

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

    ####################################################
    # interactions with other bodies and the environment
    ####################################################
    def get_interaction_point(self, xi):
        # TODO: check that it is always done using this function, never access interaction points directly!
        if not (xi in self.interaction_points.keys()):
            if (node_number := self.node_number(xi)) is not False:
                nnodes = 1
                qDOF = np.arange(self.nq_node) + self.nq_node * node_number
                uDOF = np.arange(6) + 6 * node_number

                Nq = np.eye(self.nq_node)
                Nu = np.eye(6)
                N = np.array([1.0])
            else:
                el = self.element_number(xi)
                nnodes = self.polynomial_degree + 1

                N = self.N_element(xi, el)
                Nq = np.zeros((self.nq_node, self.nq_node * nnodes))
                rows = np.arange(self.nq_node)[:, None]
                cols = rows + self.nq_node * np.arange(nnodes)
                Nq[rows, cols] = N

                Nu = np.zeros((6, 6 * nnodes))
                rows = np.arange(6)[:, None]
                cols = rows + 6 * np.arange(nnodes)
                Nu[rows, cols] = N

                # elDOF
                start = self.polynomial_degree * el
                end = self.polynomial_degree * (el + 1) + 1
                qDOF = np.arange(self.nq_node * start, self.nq_node * end)
                uDOF = np.arange(6 * start, 6 * end)

            self.interaction_points[xi] = dict(
                nnodes=nnodes,
                qDOF=qDOF,
                uDOF=uDOF,
                N=N,
                Nq=Nq,
                Nu=Nu,
                r_q=Nq[:3],
                P_q=Nq[3:],
                J_C=Nu[:3],
                B_J_R=Nu[3:],
                zero_3_nqi=np.zeros((3, self.nq_node * nnodes), dtype=float),
                zero_3_nui=np.zeros((3, 6 * nnodes), dtype=float),
                zero_3_nui_nqi=np.zeros(
                    (3, 6 * nnodes, self.nq_node * nnodes), dtype=float
                ),
            )
        return self.interaction_points[xi]

    ##########################
    # r_OP / A_IB contribution
    ##########################
    def r_OP(self, t, qi, xi, B_r_CP=zeros3):
        point_dict = self.get_interaction_point(xi)
        N = point_dict["N"]
        qnodes = qi.reshape(point_dict["nnodes"], -1)
        if B_r_CP @ B_r_CP == 0.0:
            return N @ qnodes[:, :3]

        rP = N @ qnodes
        r_CP = self._A_IB(rP[3:]) @ B_r_CP
        return rP[:3] + r_CP

    def r_OP_q(self, t, qi, xi, B_r_CP=zeros3):
        point_dict = self.get_interaction_point(xi)
        if B_r_CP @ B_r_CP == 0.0:
            return point_dict["r_q"]

        r_CP_q = np.einsum("ijk,j->ik", self.A_IB_q(t, qi, xi), B_r_CP)
        return point_dict["r_q"] + r_CP_q

    def J_P(self, t, qi, xi, B_r_CP=zeros3):
        point_dict = self.get_interaction_point(xi)
        if B_r_CP @ B_r_CP == 0.0:
            return point_dict["J_C"]

        qnodes = qi.reshape(point_dict["nnodes"], -1)
        P = point_dict["N"] @ qnodes[:, 3:]
        B_J_CP = np.cross(
            -B_r_CP[:, None], point_dict["B_J_R"], axisa=0, axisb=0, axisc=0
        )
        return point_dict["J_C"] + self._A_IB(P) @ B_J_CP

    def J_P_q(self, t, qi, xi, B_r_CP=zeros3):
        point_dict = self.get_interaction_point(xi)
        if B_r_CP @ B_r_CP == 0.0:
            return point_dict["zero_3_nui_nqi"]

        B_J_CP = np.cross(
            -B_r_CP[:, None], point_dict["B_J_R"], axisa=0, axisb=0, axisc=0
        )
        return np.einsum("ijk, jl -> ilk", self.A_IB_q(t, qi, xi), B_J_CP)

    def v_P(self, t, qi, ui, xi, B_r_CP=zeros3):
        point_dict = self.get_interaction_point(xi)
        N = point_dict["N"]
        unodes = ui.reshape(point_dict["nnodes"], -1)
        if B_r_CP @ B_r_CP == 0.0:
            return N @ unodes[:, :3]

        qnodes = qi.reshape(point_dict["nnodes"], -1)
        P = N @ qnodes[:, 3:]
        vO = N @ unodes
        B_v_CP = np.cross(vO[3:], B_r_CP)
        return vO[:3] + self._A_IB(P) @ B_v_CP

    def v_P_q(self, t, qi, ui, xi, B_r_CP=zeros3):
        point_dict = self.get_interaction_point(xi)
        if B_r_CP @ B_r_CP == 0.0:
            return point_dict["zero_3_nqi"]

        N = point_dict["N"]
        unodes = ui.reshape(point_dict["nnodes"], -1)
        B_Omega = N @ unodes[:, 3:]
        B_v_CP = np.cross(B_Omega, B_r_CP)
        return np.einsum("ijk,j->ik", self.A_IB_q(t, qi, xi), B_v_CP)

    def a_P(self, t, qi, ui, ui_dot, xi, B_r_CP=zeros3):
        point_dict = self.get_interaction_point(xi)
        N = point_dict["N"]
        u_dotnodes = ui_dot.reshape(point_dict["nnodes"], -1)
        if B_r_CP @ B_r_CP == 0.0:
            return N @ u_dotnodes[:, :3]

        qnodes = qi.reshape(point_dict["nnodes"], -1)
        unodes = ui.reshape(point_dict["nnodes"], -1)
        P = N @ qnodes[:, 3:]
        B_Omega = N @ unodes[:, 3:]
        aP = N @ u_dotnodes
        B_a_CP = np.cross(aP[3:], B_r_CP) + np.cross(B_Omega, np.cross(B_Omega, B_r_CP))
        return aP[:3] + self._A_IB(P) @ B_a_CP

    def a_P_q(self, t, qi, ui, ui_dot, xi, B_r_CP=zeros3):
        point_dict = self.get_interaction_point(xi)
        if B_r_CP @ B_r_CP == 0.0:
            return point_dict["zero_3_nqi"]

        N = point_dict["N"]
        u_dotnodes = ui_dot.reshape(point_dict["nnodes"], -1)
        unodes = ui.reshape(point_dict["nnodes"], -1)
        B_Omega = N @ unodes[:, 3:]
        aP = N @ u_dotnodes
        B_a_CP = np.cross(aP[3:], B_r_CP) + np.cross(B_Omega, np.cross(B_Omega, B_r_CP))
        return np.einsum("ijk,j->ik", self.A_IB_q(t, qi, xi), B_a_CP)

    def a_P_u(self, t, qi, ui, ui_dot, xi, B_r_CP=zeros3):
        point_dict = self.get_interaction_point(xi)
        if B_r_CP @ B_r_CP == 0.0:
            return point_dict["zero_3_nui"]

        N = point_dict["N"]
        qnodes = qi.reshape(point_dict["nnodes"], -1)
        unodes = ui.reshape(point_dict["nnodes"], -1)
        P = N @ qnodes[:, 3:]
        B_Omega = N @ unodes[:, 3:]
        B_a_CP_B_Omega = ax2skew(np.cross(B_r_CP, B_Omega)) - ax2skew(
            B_Omega
        ) @ ax2skew(B_r_CP)
        return self._A_IB(P) @ B_a_CP_B_Omega @ point_dict["Nu"][3:]

    def A_IB(self, t, qi, xi):
        point_dict = self.get_interaction_point(xi)
        N = point_dict["N"]
        qnodes = qi.reshape(point_dict["nnodes"], -1)
        return self._A_IB(N @ qnodes[:, 3:])

    def A_IB_q(self, t, qi, xi):
        point_dict = self.get_interaction_point(xi)
        N = point_dict["N"]
        qnodes = qi.reshape(point_dict["nnodes"], -1)
        P = N @ qnodes[:, 3:]
        return self._A_IB_P(P) @ point_dict["P_q"]

    def B_J_R(self, t, qi, xi):
        point_dict = self.get_interaction_point(xi)
        return point_dict["B_J_R"]

    def B_J_R_q(self, t, qi, xi):
        point_dict = self.get_interaction_point(xi)
        return point_dict["zero_3_nui_nqi"]

    def B_Omega(self, t, qi, ui, xi):
        point_dict = self.get_interaction_point(xi)
        N = point_dict["N"]
        unodes = ui.reshape(point_dict["nnodes"], -1)
        return N @ unodes[:, 3:]

    def B_Omega_q(self, t, qi, ui, xi):
        point_dict = self.get_interaction_point(xi)
        return point_dict["zero_3_nqi"]

    def B_Psi(self, t, qi, ui, ui_dot, xi):
        point_dict = self.get_interaction_point(xi)
        N = point_dict["N"]
        u_dotnodes = ui_dot.reshape(point_dict["nnodes"], -1)
        return N @ u_dotnodes[:, 3:]

    def B_Psi_q(self, t, qi, ui, ui_dot, xi):
        point_dict = self.get_interaction_point(xi)
        return point_dict["zero_3_nqi"]

    def B_Psi_u(self, t, qi, ui, ui_dot, xi):
        point_dict = self.get_interaction_point(xi)
        return point_dict["zero_3_nui"]

    #########################
    # external virtual work #
    # by distributed load   #
    #########################
    def E_pot_ext(self, t, q):
        qnodes = q.reshape(self.nnodes, -1)
        r_OC = self.N_ext @ qnodes[:, :3]
        b_qp = self.distributed_load[0](t, self.qp_ext_vec)
        return -np.einsum("ij,ij", r_OC, b_qp * self.weights_ext[:, None])

    def f_ext(self, t, q, u):
        b_qp = self.distributed_load[0](t, self.qp_ext_vec)
        B_c_qp = self.distributed_load[1](t, self.qp_ext_vec)

        f_ext = np.empty((self.nnodes, 6))
        f_ext[:, :3] = self.N_ext.T @ (b_qp * self.weights_ext[:, None])
        f_ext[:, 3:] = self.N_ext.T @ (B_c_qp * self.weights_ext[:, None])
        return f_ext.reshape(-1)

    #########################
    # internal virtual work #
    #########################
    def l_sigma(self, q):
        _, B_gamma_bar, B_kappa_bar = self._eval_internal_vec(
            self.N_int, self.N_xi_int, q
        )

        epsilon_bar = np.empty((self.nquadrature_int_total, 6))
        epsilon_bar[:, :3] = B_gamma_bar - self.B_gamma0_bar_int
        epsilon_bar[:, 3:] = B_kappa_bar - self.B_kappa0_bar_int

        l = self.Nc_int.T @ (epsilon_bar * self.qw_int_vec[:, None])

        l_c = l[:, self.idx_c].reshape(-1)
        l_g = l[:, self.idx_g].reshape(-1)
        return l_c, l_g

    def l_sigma_q(self, q):
        # compute W_sigma
        _eval, _deval = self._eval_internal_vec(
            self.N_int, self.N_xi_int, q, deval=True
        )
        A_IB = _eval[0]
        T, B_gamma_bar_P, B_kappa_bar_P = _deval

        # TODO: make sparse?
        # c_sigma_q_qp[N/N_xi, qpi, la_cDOF, qDOF]
        c_sigma_q_qp = np.empty((2, self.nquadrature_int_total, 6, self.nq_node))
        # to be multiplied with N_xi
        c_sigma_q_qp[1, :, :3, :3] = A_IB.transpose((0, 2, 1))
        c_sigma_q_qp[1, :, :3, 3:] = 0.0
        c_sigma_q_qp[1, :, 3:, :3] = 0.0
        c_sigma_q_qp[1, :, 3:, 3:] = T

        # to be multiplied with N
        c_sigma_q_qp[0, :, :, :3] = 0.0
        c_sigma_q_qp[0, :, :3, 3:] = B_gamma_bar_P
        c_sigma_q_qp[0, :, 3:, 3:] = B_kappa_bar_P

        return self.c_sigma_q_SAB.add_blocks(c_sigma_q_qp)

    def l_sigma_dot_q(self, q, u):
        # TODO: do this without approx_fprime
        l_c_dot_q = approx_fprime(q, lambda q_: self.W_sigma(q_)[0].T @ u)
        l_g_dot_q = approx_fprime(q, lambda q_: self.W_sigma(q_)[1].T @ u)
        return l_c_dot_q, l_g_dot_q

    def l_sigma_ddot(self, q, u, u_dot):
        # TODO: do this consistent, i.e., w/o q_dot
        W_c, W_g = self.W_sigma(q)
        l_c_dot_q, l_g_dot_q = self.l_sigma_dot_q(q, u)
        q_dot = self.q_dot(0.0, q, u)

        l_c_ddot = W_c.T @ u_dot + l_c_dot_q @ q_dot
        l_g_ddot = W_g.T @ u_dot + l_g_dot_q @ q_dot
        return l_c_ddot, l_g_ddot

    def _c_la_c_coo(self):
        C_qp = self.material_model.C_inv(self.qp_int_vec, quadrature=True)
        c_la_c = self.c_la_c_SAB.add_blocks(
            C_qp[None, :, self.idx_c[:, None], self.idx_c]
        )

        self._cla_c = c_la_c
        self._cla_c_inv = spsolve(c_la_c.tocsc(), eye_array(self.nla_c, format="csc"))
        return c_la_c

    def W_sigma(self, q):
        # compute W_sigma
        A_IB, B_gamma_bar, B_kappa_bar = self._eval_internal_vec(
            self.N_int, self.N_xi_int, q
        )

        gamma_bar_tilde = ax2skew(B_gamma_bar)
        kappa_bar_tilde = ax2skew(B_kappa_bar)

        # W_sigma_qp[N/N_xi, qpi, uDOF, la_cDOF]
        W_sigma_qp = np.empty((2, self.nquadrature_int_total, 6, 6))
        # to be multiplied with N_xi
        W_sigma_qp[1, :, :3, :3] = -A_IB
        W_sigma_qp[1, :, :3, 3:] = 0.0
        W_sigma_qp[1, :, 3:, :3] = 0.0
        W_sigma_qp[1, :, 3:, 3:] = -eye3

        # to be multiplied with N
        W_sigma_qp[0, :, :3, :] = 0.0
        W_sigma_qp[0, :, 3:, :3] = gamma_bar_tilde
        W_sigma_qp[0, :, 3:, 3:] = kappa_bar_tilde

        return self.W_sigma_SAB.add_blocks(W_sigma_qp)

    def Wla_sigma(self, q, la_c=None, la_g=None):
        la_sigma_nodes = np.zeros((self.nnodes_sigma, 6))
        if la_c is not None:
            la_sigma_nodes[:, self.idx_c] = la_c.reshape(self.nnodes_sigma, -1)
        if la_g is not None:
            la_sigma_nodes[:, self.idx_g] = la_g.reshape(self.nnodes_sigma, -1)
        _eval = self._eval_internal_vec(self.N_int, self.N_xi_int, q)
        return self.h_pot(_eval, self.Nc_int @ la_sigma_nodes)

    def h_pot(self, _eval, sigma_qp):
        # compute generalized internal forces based on evaluation and forces at quadrature points
        A_IB, B_gamma_bar, B_kappa_bar = _eval

        # h_pot_qp[N/N_xi, qpi, uDOF]
        h_pot_qp = np.empty((2, self.nquadrature_int_total, 6))
        # to be multiplied with N_xi
        h_pot_qp[1, :, :3] = -np.einsum("ijk,ik->ij", A_IB, sigma_qp[:, :3])
        h_pot_qp[1, :, 3:] = -sigma_qp[:, 3:]

        # to be multiplied with N
        h_pot_qp[0, :, :3] = 0.0
        h_pot_qp[0, :, 3:] = np.cross(B_gamma_bar, sigma_qp[:, :3]) + np.cross(
            B_kappa_bar, sigma_qp[:, 3:]
        )

        # add together and multiply with quadrature weights
        h_pot_nodes = self.N_int.T @ (
            h_pot_qp[0] * self.qw_int_vec[:, None]
        ) + self.N_xi_int.T @ (h_pot_qp[1] * self.qw_int_vec[:, None])
        return h_pot_nodes.reshape(-1)

    def Wla_sigma_q(self, q, la_c=None, la_g=None):
        la_sigma_nodes = np.zeros((self.nnodes_sigma, 6))
        if la_c is not None:
            la_sigma_nodes[:, self.idx_c] = la_c.reshape(self.nnodes_sigma, -1)
        if la_g is not None:
            la_sigma_nodes[:, self.idx_g] = la_g.reshape(self.nnodes_sigma, -1)
        sigma_qp = self.Nc_int @ la_sigma_nodes

        _eval, _deval = self._eval_internal_vec(
            self.N_int, self.N_xi_int, q, deval=True
        )
        A_IB = _eval[0]
        T, B_gamma_bar_P, B_kappa_bar_P = _deval
        I_n_P = np.einsum(
            "ijk,ikl->ijl",
            A_IB,
            np.cross(sigma_qp[:, :3, None], T, axisa=1, axisb=1, axisc=1),
        )

        # TODO: make sparse?
        # Wla_sigma_qp_qbar[N/N_xi, qpi, uDOF, qDOF]
        Wla_sigma_qp_qbar = np.zeros((4, self.nquadrature_int_total, 6, self.nq_node))
        # to be multiplied with N_xi <-> N
        Wla_sigma_qp_qbar[2, :, :3, 3:] = I_n_P

        # to be multiplied with N <-> N_xi
        Wla_sigma_qp_qbar[1, :, 3:, :3] = -np.cross(
            sigma_qp[:, :3, None], A_IB, axisa=1, axisb=2, axisc=1
        )  # A_IB.T in gamma -> axisb=2
        Wla_sigma_qp_qbar[1, :, 3:, 3:] = -np.cross(
            sigma_qp[:, 3:, None], T, axisa=1, axisb=1, axisc=1
        )

        # to be multiplied with N <-> N
        Wla_sigma_qp_qbar[0, :, 3:, 3:] = -(
            np.cross(sigma_qp[:, :3, None], B_gamma_bar_P, axisa=1, axisb=1, axisc=1)
            + np.cross(sigma_qp[:, 3:, None], B_kappa_bar_P, axisa=1, axisb=1, axisc=1)
        )
        return self.h_pot_q_SAB.add_blocks(Wla_sigma_qp_qbar)

    def f_pot(self, t, q, u):
        _eval = self._eval_internal_vec(self.N_int, self.N_xi_int, q)
        epsilon = np.hstack([_eval[1], _eval[2]]) / self.J_int_vec[:, None]
        sigma_db = self.material_model.sigma(
            self.qp_int_vec, epsilon, self.epsilon0_int, quadrature=True
        )

        sigma_qp = np.zeros((self.nquadrature_int_total, 6))
        sigma_qp[:, self.idx_db] = sigma_db[:, self.idx_db]

        return self.h_pot(_eval, sigma_qp)

    def f_pot_q(self, t, q, u):
        _eval, _deval = self._eval_internal_vec(
            self.N_int, self.N_xi_int, q, deval=True
        )
        epsilon = np.hstack([_eval[1], _eval[2]]) / self.J_int_vec[:, None]
        sigma_db = self.material_model.sigma(
            self.qp_int_vec, epsilon, self.epsilon0_int, quadrature=True
        )

        sigma_qp = np.zeros((self.nquadrature_int_total, 6))
        sigma_qp[:, self.idx_db] = sigma_db[:, self.idx_db]

        A_IB, B_gamma_bar, B_kappa_bar = _eval
        T, B_gamma_bar_P, B_kappa_bar_P = _deval

        ######################
        # material stiffness #
        ######################
        A_IB_transpose_to_J = A_IB.transpose(0, 2, 1) / self.J_int_vec[:, None, None]
        T_to_J = T / self.J_int_vec[:, None, None]

        sigma_epsilon = self.material_model.sigma_epsilon(
            self.qp_int_vec, epsilon, self.epsilon0_int, quadrature=True
        )
        B_n_gamma, B_n_kappa, B_m_gamma, B_m_kappa = sigma_epsilon

        # fmt: off
        B_n_P = (B_n_gamma @ B_gamma_bar_P + B_n_kappa @ B_kappa_bar_P) / self.J_int_vec[:, None, None]
        B_m_P = (B_m_gamma @ B_gamma_bar_P + B_m_kappa @ B_kappa_bar_P) / self.J_int_vec[:, None, None]
        # fmt: on

        # TODO: make sparse?
        # f_pot_qp_qbar[N/N_xi, qpi, uDOF, qDOF]
        f_pot_qp_qbar = np.zeros((4, self.nquadrature_int_total, 6, self.nq_node))
        # to be multiplied with N_xi <-> N_xi
        f_pot_qp_qbar[3, :, :3, :3] = -A_IB @ B_n_gamma @ A_IB_transpose_to_J
        f_pot_qp_qbar[3, :, :3, 3:] = -A_IB @ B_n_kappa @ T_to_J
        f_pot_qp_qbar[3, :, 3:, :3] = -B_m_gamma @ A_IB_transpose_to_J
        f_pot_qp_qbar[3, :, 3:, 3:] = -B_m_kappa @ T_to_J

        # to be multiplied with N_xi <-> N
        f_pot_qp_qbar[2, :, :3, 3:] = -A_IB @ B_n_P
        f_pot_qp_qbar[2, :, 3:, 3:] = -B_m_P

        # to be multiplied with N <-> N_xi
        f_pot_qp_qbar[1, :, 3:, :3] = (
            np.cross(B_gamma_bar[:, :, None], B_n_gamma, axisa=1, axisb=1, axisc=1)
            + np.cross(B_kappa_bar[:, :, None], B_m_gamma, axisa=1, axisb=1, axisc=1)
        ) @ A_IB_transpose_to_J
        f_pot_qp_qbar[1, :, 3:, 3:] = (
            np.cross(B_gamma_bar[:, :, None], B_n_kappa, axisa=1, axisb=1, axisc=1)
            + np.cross(B_kappa_bar[:, :, None], B_m_kappa, axisa=1, axisb=1, axisc=1)
        ) @ T_to_J

        # to be multiplied with N <-> N
        f_pot_qp_qbar[0, :, 3:, 3:] = np.cross(
            B_gamma_bar[:, :, None], B_n_P, axisa=1, axisb=1, axisc=1
        ) + np.cross(B_kappa_bar[:, :, None], B_m_P, axisa=1, axisb=1, axisc=1)

        ##################
        # geometric part #
        ##################
        # TODO: can we use this from Wla_sigma_q?
        I_n_P = np.einsum(
            "ijk,ikl->ijl",
            A_IB,
            np.cross(sigma_qp[:, :3, None], T, axisa=1, axisb=1, axisc=1),
        )

        # to be multiplied with N_xi <-> N
        f_pot_qp_qbar[2, :, :3, 3:] += I_n_P

        # to be multiplied with N <-> N_xi
        f_pot_qp_qbar[1, :, 3:, :3] -= np.cross(
            sigma_qp[:, :3, None], A_IB, axisa=1, axisb=2, axisc=1
        )  # A_IB.T in gamma -> axisb=2
        f_pot_qp_qbar[1, :, 3:, 3:] -= np.cross(
            sigma_qp[:, 3:, None], T, axisa=1, axisb=1, axisc=1
        )

        # to be multiplied with N <-> N
        f_pot_qp_qbar[0, :, 3:, 3:] -= np.cross(
            sigma_qp[:, :3, None], B_gamma_bar_P, axisa=1, axisb=1, axisc=1
        ) + np.cross(sigma_qp[:, 3:, None], B_kappa_bar_P, axisa=1, axisb=1, axisc=1)

        return self.h_pot_q_SAB.add_blocks(f_pot_qp_qbar)

    ########################
    # evaluation functions #
    ########################
    def eval_stresses(self, t, q, la_c, la_g, n_per_element=None, n_ges=None):
        xis, els = self._eval_logic(n_per_element, n_ges)

        # TODO: are there problems, if everything is DB?
        # stresses due to compliance and constraints
        Nc = self.Nc(xis, els)
        la_sigma_nodes = np.zeros((self.nnodes_sigma, 6))
        if self._nla_c > 0:
            la_sigma_nodes[:, self.idx_c] = la_c[self.la_cDOF].reshape(
                self.nnodes_sigma, -1
            )
        if self._nla_g > 0:
            la_sigma_nodes[:, self.idx_g] = la_g[self.la_gDOF].reshape(
                self.nnodes_sigma, -1
            )
        sigma = Nc @ la_sigma_nodes

        # stresses due to displacement based
        if self._nDB > 0:
            N, N_xi = self.N(xis, els)
            # reference strains
            _, B_gamma0_bar, B_kappa0_bar = self._eval_internal_vec(N, N_xi, self.Q)
            J = np.linalg.norm(B_gamma0_bar, axis=1)
            # current strains
            _, B_gamma_bar, B_kappa_bar = self._eval_internal_vec(N, N_xi, q[self.qDOF])

            epsilon = np.hstack([B_gamma_bar, B_kappa_bar]) / J[:, None]
            epsilon0 = np.hstack([B_gamma0_bar, B_kappa0_bar]) / J[:, None]
            sigma_db = self.material_model.sigma(
                xis, epsilon, epsilon0, quadrature=False
            )
            sigma[:, self.idx_db] = sigma_db[:, self.idx_db]

        return xis, sigma[:, :3], sigma[:, 3:]

    def eval_strains(self, t, q, la_c, la_g, n_per_element=None, n_ges=None):
        xis, els = self._eval_logic(n_per_element, n_ges)

        epsilon = np.zeros((len(xis), 6))
        if self._nDB > 0:
            N, N_xi = self.N(xis, els)
            # reference strains
            _, B_gamma0_bar, B_kappa0_bar = self._eval_internal_vec(N, N_xi, self.Q)
            J = np.linalg.norm(B_gamma0_bar, axis=1)
            # current strains
            _, B_gamma_bar, B_kappa_bar = self._eval_internal_vec(N, N_xi, q[self.qDOF])

            epsilon[:, :3] = (B_gamma_bar - B_gamma0_bar) / J[:, None]
            epsilon[:, 3:] = (B_kappa_bar - B_kappa0_bar) / J[:, None]

        # strains from compliance
        if self._nla_c > 0:
            Nc = self.Nc(xis, els)
            la_c_nodes = la_c[self.la_cDOF].reshape(self.nnodes_sigma, -1)
            la_sigma = Nc @ la_c_nodes

            C_inv = self.material_model.C_inv(xis, quadrature=False)
            epsilon[:, self.idx_c] = np.einsum("ijk,ik->ij", C_inv, la_sigma)

        # strains from constraints are always 0
        epsilon[:, self.idx_g] = 0.0

        return xis, epsilon[:, :3], epsilon[:, 3:]


def make_BoostedCosseratRod(
    *,
    polynomial_degree=None,
    idx_constraints=None,
    idx_displacement_based=None,
    quadrature_int=None,
    quadrature_dyn=None,
    quadrature_ext=None,
    parametrization=None,
) -> RodInterface:
    """idx_constraints and idx_displacement_based in [0, 1, 2, 3, 4, 5]"""
    # polynomila degree
    polynomial_degree = 2 if polynomial_degree is None else polynomial_degree

    # constraints
    if idx_constraints is not None:
        idx_constraints = np.asarray(idx_constraints, dtype=int)
        if not ((idx_constraints >= 0).all() & (idx_constraints <= 5).all()):
            raise ValueError("constraint values must between 0 and 5")
    else:
        idx_constraints = np.array([], dtype=int)
    idx_constraints = np.sort(idx_constraints)

    # displacement based
    if idx_displacement_based is not None:
        idx_displacement_based = np.asarray(idx_displacement_based, dtype=int)
        if not (
            (idx_displacement_based >= 0).all() & (idx_displacement_based <= 5).all()
        ):
            raise ValueError("displacement_based values must between 0 and 5")
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
        quadrature_int = (polynomial_degree, "Gauss")
    elif isinstance(quadrature_int, int):
        quadrature_int = (quadrature_int, "Gauss")
    elif not isinstance(quadrature_int, tuple):
        raise ValueError(
            "quadrature_int must be either an 'None', an integer for Gauss quadrature or a tuple: (nquadrature, method)"
        )

    if quadrature_dyn == None:
        # TODO: take trapezoidal rule as default?
        quadrature_dyn = (polynomial_degree + 1, "Trapezoidal")
        n_full = int(np.ceil(3 / 2 * polynomial_degree + 1 / 2))
        quadrature_dyn = (n_full, "Gauss")
        print(f"quadrature_dyn: {quadrature_dyn}")
    elif isinstance(quadrature_dyn, int):
        quadrature_dyn = (quadrature_dyn, "Gauss")
    elif not isinstance(quadrature_dyn, tuple):
        raise ValueError(
            "quadrature_dyn must be either an 'None', an integer for Gauss quadrature or a tuple: (nquadrature, method)"
        )

    if quadrature_ext == None:
        # TODO: take trapezoidal rule as default?
        quadrature_ext = (polynomial_degree + 1, "Trapezoidal")
        n_full = int(np.ceil(3 / 2 * polynomial_degree + 1 / 2))
        quadrature_ext = (n_full, "Gauss")
        print(f"quadrature_ext: {quadrature_ext}")
    elif isinstance(quadrature_ext, int):
        quadrature_ext = (quadrature_ext, "Gauss")
    elif not isinstance(quadrature_ext, tuple):
        raise ValueError(
            "quadrature_ext must be either an 'None', an integer for Gauss quadrature or a tuple: (nquadrature, method)"
        )

    # parametrization
    if parametrization is None:
        parametrization = "Quaternion"

    assert parametrization in [
        "Quaternion",
        "R12",
    ], f"parametrization {parametrization} is not supported!"

    if parametrization == "Quaternion":
        _Log_A_IB = Log_SO3_quat
        _A_IB = Exp_SO3_quat
        _A_IB_P = Exp_SO3_quat_P
        _T_IB = T_SO3_quat
        _T_IB_P = T_SO3_quat_P
        _T_IB_inv = T_SO3_inv_quat
        _T_IB_inv_P = T_SO3_inv_quat_P
        nq_node = 7
    elif parametrization == "R12":
        _Log_A_IB = Log_SO3_R9
        _A_IB = Exp_SO3_R9
        _A_IB_P = Exp_SO3_R9_R9
        _T_IB = T_SO3_R9
        _T_IB_P = T_SO3_R9_R9
        _T_IB_inv = T_SO3_inv_R9
        _T_IB_inv_P = T_SO3_inv_R9_R9
        nq_node = 12

    class BoostedCosseratRod_PetrovGalerkin(CosseratRod_PetrovGalerkin):
        def _pre_init_(self):
            # functions for orientation
            self._A_IB = _A_IB
            self._A_IB_P = _A_IB_P
            self._T_IB = _T_IB
            self._T_IB_P = _T_IB_P
            self._T_IB_inv = _T_IB_inv
            self._T_IB_inv_P = _T_IB_inv_P(None)  # evaluate as it is constant
            self.nq_node = nq_node

            self.idx_g = idx_constraints
            self.idx_db = idx_displacement_based
            self.idx_c = np.setdiff1d(
                np.arange(6), np.union1d(idx_constraints, idx_displacement_based)
            )
            self.polynomial_degree = polynomial_degree

            self.quadrature_int = quadrature_int
            self.quadrature_dyn = quadrature_dyn
            self.quadrature_ext = quadrature_ext

            # Quaternion: unit quaternion constraints
            # Directors: unit-length and orthogonality constriant
            self.dim_g_S = self.nq_node - 6

        if parametrization == "Quaternion":

            def g_S(self, t, q):
                qnodes = q.reshape(self.nnodes, -1)
                return np.sum(qnodes[:, 3:] ** 2, axis=1) - 1

            def g_S_q(self, t, q):
                qnodes = q.reshape(self.nnodes, -1)
                coo = CooMatrix((self.nla_S, self.nq))
                coo.data = 2 * qnodes[:, 3:].reshape(-1)
                coo.row = self._g_S_q_row
                coo.col = self._g_S_q_col
                return coo

            def step_callback(self, t, q, u):
                # TODO: step_callback with R12? Using spurrier and Exp_SO3_quat?
                """ "Quaternion normalization after each time step."""
                qnodes = q.reshape(self.nnodes, -1)
                qnodes[:, 3:] /= np.linalg.norm(qnodes[:, 3:], axis=1)[:, None]
                # Note: qnodes shares still memory with q here
                return qnodes.reshape(-1), u

        elif parametrization == "R12":

            def g_S(self, t, q):
                # TODO
                return np.zeros(self.nla_S)

            def g_S_q(self, t, q):
                # TODO
                return np.zeros((self.nla_S, self.nq))

            # TODO: step_callback?

        @staticmethod
        def straight_configuration(
            nelement,
            L,
            r_OP0=zeros3,
            A_IB0=eye3,
        ):
            """Compute generalized position coordinates for straight configuration."""
            nnodes = polynomial_degree * nelement + 1

            r_OP = np.zeros((3, nnodes))
            r_OP[0] = np.linspace(0, L, num=nnodes)
            P = _Log_A_IB(A_IB0)
            rP = np.zeros((nnodes, nq_node), dtype=float)
            for i in range(nnodes):
                rP[i, :3] = r_OP0 + A_IB0 @ r_OP[:, i]
                rP[i, 3:] = P

            return rP.reshape(-1)

        @staticmethod
        def pose_configuration(
            nelement,
            r_OP,
            A_IB,
            xi1=1.0,
            r_OP0=zeros3,
            A_IB0=eye3,
        ):
            """Compute generalized position coordinates for a pre-curved rod with centerline curve r_OP and orientation of A_IB."""
            assert callable(r_OP), "r_OP must be callable!"
            assert callable(A_IB), "A_IB must be callable!"

            nnodes = polynomial_degree * nelement + 1
            xis = np.linspace(0, xi1, nnodes)

            # nodal positions and unit quaternions
            rP = np.zeros((nnodes, nq_node))
            for i, xii in enumerate(xis):
                rP[i, :3] = r_OP0 + A_IB0 @ r_OP(xii)
                A_IBi = A_IB0 @ A_IB(xii)
                rP[i, 3:] = _Log_A_IB(A_IBi)

            # check for the right quaternion hemisphere
            for i in range(nnodes - 1):
                inner = rP[i, 3:] @ rP[i + 1, 3:]
                if inner < 0:
                    rP[i + 1, 3:] *= -1

            return rP.reshape(-1)

        @staticmethod
        def straight_initial_configuration(
            nelement, L, r_OP0=zeros3, A_IB0=eye3, v_P0=zeros3, B_omega_IB0=eye3
        ): ...

    return BoostedCosseratRod_PetrovGalerkin
