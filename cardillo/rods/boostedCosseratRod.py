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
    Exp_SO3_quat,
    Exp_SO3_quat_P,
    Log_SO3_quat,
    T_SO3_quat,
    T_SO3_quat_P,
    T_SO3_quat_Q_P,
    T_SO3_inv_quat,
    T_SO3_inv_quat_P,
)
from cardillo.utility.coo_matrix import CooMatrix
from cardillo.utility.sparse_array_blocks import SparseArrayBlocks


from ._base_export import RodExportBase
from ._cross_section import CrossSectionInertias
from .discretization.mesh1D import Mesh1D_equidistant

zeros3 = np.zeros(3, dtype=float)
eye3 = np.eye(3, dtype=float)


class CosseratRod_PetrovGalerkin(RodExportBase):
    def __init__(
        self,
        cross_section,
        material_model,
        cross_section_inertias,
        idx_constraints,
        idx_displacement_based,
        nelement,
        polynomial_degree,
        nquadrature_int,
        nquadrature_dyn,
        Q,
        q0,
        u0,
        name,
    ):
        # call base class for all export properties
        super().__init__(cross_section)

        # rod properties
        self.material_model = material_model
        if cross_section_inertias == None:
            self.cross_section_inertias = CrossSectionInertias()
        else:
            #
            self.cross_section_inertias = cross_section_inertias
            self.h = self._h
            self.h_u = self._h_u

        self.idx_g = idx_constraints
        self.idx_db = idx_displacement_based
        self.idx_c = np.setdiff1d(
            np.arange(6), np.union1d(idx_constraints, idx_displacement_based)
        )

        self.name = "Cosserat_rod" if name is None else name

        self.nelement = nelement
        self.polynomial_degree = polynomial_degree

        mesh_kin = Mesh1D_equidistant(
            basis="Lagrange",
            nelement=nelement,
            polynomial_degere=polynomial_degree,
            derivative_order=1,
        )

        # total number of nodes and per element
        self.nnodes = mesh_kin.nnodes
        self.nnodes_element = mesh_kin.nnodes_element

        self.xis_nodes = mesh_kin.xis_nodes
        self.xis_element_boundaries = mesh_kin.xis_element

        # total number of generalized position and velocity coordinates
        self.nq = self.nnodes * 7
        self.nu = self.nnodes * 6

        self.N = lambda xis, els: mesh_kin.shape_functions(xis, els, 0)[0]
        self.N_element = lambda xi, el: mesh_kin.shape_function_array_element(xi, el, 0)

        #####################
        # quadrature points #
        #####################
        quadrature_int_kin = mesh_kin.quadrature(nquadrature_int, "Gauss", 1)
        self.nquadrature_int_total = quadrature_int_kin["nquadrature_total"]
        self.qp_int_vec = quadrature_int_kin["qp"]
        self.qw_int_vec = quadrature_int_kin["qw"]
        self.qels_int_vec = quadrature_int_kin["els"]
        self.N_int, self.N_xi_int = quadrature_int_kin["N"]

        # TODO: allow for trapezoidal rule
        quadrature_dyn_kin = mesh_kin.quadrature(nquadrature_dyn, "Gauss", 1)
        self.nquadrature_dyn_total = quadrature_dyn_kin["nquadrature_total"]
        self.qp_dyn_vec = quadrature_dyn_kin["qp"]
        self.qw_dyn_vec = quadrature_dyn_kin["qw"]
        self.qels_dyn_vec = quadrature_dyn_kin["els"]
        self.N_dyn, self.N_xi_dyn = quadrature_dyn_kin["N"]

        # referential generalized position coordinates, initial generalized
        # position and velocity coordinates
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else u0

        # unit quaternion constraints
        dim_g_S = 1
        self.nla_S = self.nnodes * dim_g_S
        self.la_S0 = np.zeros(self.nla_S, dtype=float)
        self._g_S_q_row = np.repeat(np.arange(self.nnodes), 4)
        self._g_S_q_col = (
            (3 + 7 * np.arange(self.nnodes))[:, None] + np.arange(4)
        ).ravel()

        ###############################
        # compliance and constrtaints #
        ###############################
        mesh_cg = Mesh1D_equidistant(
            basis="Lagrange_Disc",
            nelement=nelement,
            polynomial_degere=polynomial_degree - 1,
            derivative_order=0,
        )

        # total number of nodes
        self.nnodes_sigma = mesh_cg.nnodes

        # total number of compliance coordinates
        self.nla_sigma = self.nnodes_sigma * 6
        if (nla_c := self.nnodes_sigma * len(self.idx_c)) > 0:
            self.nla_c = self._nla_c = nla_c

            ##############
            # compliance #
            # c = c_la_c @ la_c - l_c
            ##############
            # la_c = -c_la_c @ c(q, u, 0)
            self.la_c = lambda t, q, u: self.__cla_c_inv @ self.l_sigma(q, u)[0]
            self.c = lambda t, q, u, la_c: self.__cla_c @ la_c - self.l_sigma(q, u)[0]
            self.c_q = lambda t, q, u, la_c: -self.l_sigma_q(q, u)[0]
            self.c_la_c = lambda: self.__cla_c
            self.W_c = lambda t, q: self.W_sigma(q)[0]
            self.Wla_c_q = lambda t, q, la_c: self.Wla_sigma_q(q, la_c, None)
        else:
            self._nla_c = 0

        if (nla_g := self.nnodes_sigma * len(self.idx_g)) > 0:
            self.nla_g = self._nla_g = nla_g

            #############
            # constraints
            #  g = - l_g
            #############
            self.nu_zeros = np.zeros(self.nu)
            self.g = lambda t, q: -self.l_sigma(q, self.nu_zeros)[1]
            self.g_q = lambda t, q: -self.l_sigma_q(q, self.nu_zeros)[1]
            self.W_g = lambda t, q: self.W_sigma(q)[1]
            self.Wla_g_q = lambda t, q, la_g: self.Wla_sigma_q(q, None, la_g)

            # TODO:
            self.g_dot = lambda t, q, u: self.W_sigma(q)[1].T @ u
            self.g_dot_u = lambda t, q: self.W_sigma(q)[1].T
            self.g_dot_q = lambda t, q, u: ...
            self.g_ddot = lambda t, q, u, u_dot: np.zeros(self.nla_g)

        else:
            self._nla_g = 0

        if len(self.idx_db) > 0:
            # TODO: add h and h_q DB
            raise NotImplementedError

        self.Nc = lambda xis, els: mesh_cg.shape_functions(xis, els, 0)[0]
        self.Nc_element = lambda xi, el: mesh_cg.shape_function_array_element(xi, el, 0)

        quadrature_int_cg = mesh_cg.quadrature(nquadrature_int, "Gauss", 0)
        self.Nc_int = quadrature_int_cg["N"][0]

        # TODO: do inner dict with enum as key or use a class instead of dict
        self.interaction_points: dict[float, dict[str, np.ndarray]] = {}

        # reference strains for weight of blocks
        self.set_reference_strains(Q)
        ################
        # assemble matrices for block structure
        ################
        # M
        M_pairs = [(self.N_dyn, self.N_dyn, self.qw_dyn_vec * self.J_dyn_vec)]
        self.M_h_u_SAB = SparseArrayBlocks((self.nu, self.nu), (6, 6), M_pairs)

        # c_la_c
        c_la_c_pairs = [(self.Nc_int, self.Nc_int, self.qw_int_vec * self.J_int_vec)]
        self.c_la_c_SAB = SparseArrayBlocks(
            (self.nla_c, self.nla_c), (len(self.idx_c), len(self.idx_c)), c_la_c_pairs
        )

        # c_q
        c_q_pairs = [
            (self.Nc_int, self.N_int, self.qw_int_vec),
            (self.Nc_int, self.N_xi_int, self.qw_int_vec),
        ]
        self.c_sigma_q_SAB = SparseArrayBlocks(
            (self.nla_sigma, self.nq),
            (6, 7),
            c_q_pairs,
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
        Wla_sigma_q_pairs = [
            (self.N_int, self.N_int, self.qw_int_vec),
            (self.N_int, self.N_xi_int, self.qw_int_vec),
            (self.N_xi_int, self.N_int, self.qw_int_vec),
        ]
        self.Wla_sigma_q_SAB = SparseArrayBlocks(
            (self.nu, self.nq), (6, 7), Wla_sigma_q_pairs
        )

    def _eval_internal_vec(self, N, N_xi, q, deval=False):
        qbar_nodes = q.reshape(self.nnodes, -1)
        P_IB = N @ qbar_nodes[:, 3:]
        qbar_xi = N_xi @ qbar_nodes

        A_IB = self._A_IB(P_IB)
        T = self._T_IB(P_IB)

        B_gamma_bar = np.einsum("ijk,ij->ik", A_IB, qbar_xi[:, :3])
        if not deval:
            B_kappa_bar = np.einsum("ijk,ik->ij", T, qbar_xi[:, 3:])
            return A_IB, B_gamma_bar, B_kappa_bar

        # using my magic property
        B_gamma_bar_P = np.cross(B_gamma_bar[:, :, None], T, axisa=1, axisb=1, axisc=1)

        T_IB_P = self._T_IB_P(P_IB)
        B_kappa_bar_P = np.einsum("ijkl,ik->ijl", T_IB_P, qbar_xi[:, 3:])
        return A_IB, T, B_gamma_bar_P, B_kappa_bar_P

    def set_reference_strains(self, Q):
        self.Q = Q.copy()

        _, B_gamma_bar, B_kappa_bar = self._eval_internal_vec(
            self.N_int, self.N_xi_int, self.Q
        )
        self.J_int_vec = np.linalg.norm(B_gamma_bar, axis=1)
        self.B_gamma0_bar = B_gamma_bar
        self.B_kappa0_bar = B_kappa_bar

        _, B_gamma_bar, _ = self._eval_internal_vec(self.N_dyn, self.N_xi_dyn, self.Q)
        self.J_dyn_vec = np.linalg.norm(B_gamma_bar, axis=1)

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

    # def frames(self, qsystem, num=10):
    #     qbody = qsystem[self.qDOF]
    #     qnodes = qbody.reshape(self.nnodes, -1)

    #     # maybe cache the N matrix
    #     # TODO: update on how to get N
    #     N = self.mesh_kin.N(np.linspace(0, 1, num=num))
    #     qpoints = N @ qnodes
    #     rs = qpoints[:, :3].T
    #     A_IBs = self._A_IB(qpoints[:, 3:])
    #     return rs, A_IBs[:, :, 0].T, A_IBs[:, :, 1].T, A_IBs[:, :, 2].T

    ##################
    # abstract methods
    ##################
    def assembler_callback(self):
        self._M_coo()
        self._c_la_c_coo()

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        qnodes = q.reshape(self.nnodes, 7)
        unodes = u.reshape(self.nnodes, 6)

        qnodes_dot = np.empty((self.nnodes, 7), dtype=np.common_type(q, u))
        qnodes_dot[:, :3] = unodes[:, :3]
        qnodes_dot[:, 3:] = np.einsum(
            "ijk,ik->ij", T_SO3_inv_quat(qnodes[:, 3:]), unodes[:, 3:]
        )

        return qnodes_dot.reshape(-1)

    def q_dot_q(self, t, q, u):
        qnodes = q.reshape(self.nnodes, 7)
        unodes = u.reshape(self.nnodes, 6)

        blocks = np.empty((self.nnodes, 7, 7))
        blocks[:, :3] = 0.0
        blocks[:, 3:, :3] = 0.0
        blocks[:, 3:, 3:] = np.einsum(
            "ijkl,ik->ijl",
            T_SO3_inv_quat_P(qnodes[:, 3:])[None, :, :, :],
            unodes[:, 3:],
        )

        return csr_array(block_diag(blocks))

    def q_dot_u(self, t, q):
        qnodes = q.reshape(self.nnodes, 7)

        blocks = np.empty((self.nnodes, 7, 6))
        blocks[:, :3, :3] = eye3
        blocks[:, :3, 3:] = 0.0
        blocks[:, 3:, :3] = 0.0
        blocks[:, 3:, 3:] = T_SO3_inv_quat(qnodes[:, 3:])

        return csr_array(block_diag(blocks))  # this keeps the 0's

    def step_callback(self, t, q, u):
        """ "Quaternion normalization after each time step."""
        qnodes = q.reshape(self.nnodes, -1)
        qnodes[:, 3:] /= np.linalg.norm(qnodes[:, 3:], axis=1)[:, None]
        # Note: qnodes shares still memory with q here
        return qnodes.reshape(-1), u

    ###################################
    # TODO: Add energies and momenta? #
    ###################################

    #########################################
    # equations of motion
    #########################################
    def M(self, t, q):
        return self.__M

    def _M_coo(self):
        self.constant_mass_matrix = True

        # TODO: make this sparse?
        M_qp = np.empty((1, self.nquadrature_dyn_total, 6, 6))
        M_qp[0, :, :3, :3] = (
            eye3 * self.cross_section_inertias.A_rho0
        )  # (self.qp_dyn_vec)
        M_qp[0, :, :3, 3:] = 0.0
        M_qp[0, :, 3:, :3] = 0.0
        M_qp[0, :, 3:, 3:] = self.cross_section_inertias.B_I_rho0  # (self.qp_dyn_vec)

        self.__M = self.M_h_u_SAB.add_blocks(M_qp)
        return self.__M

    def _h(self, t, q, u):
        unodes = u.reshape(self.nnodes, -1)
        B_Omega = self.N_dyn @ unodes[:, 3:]

        # spin
        B_L = B_Omega @ self.cross_section_inertias.B_I_rho0.T
        f_gyr_qp = np.cross(B_Omega, B_L)

        f_gyr = np.empty((self.nnodes, 6))
        f_gyr[:, :3] = 0.0
        f_gyr[:, 3:] = self.N_dyn.T @ (
            f_gyr_qp * (-self.J_dyn_vec * self.qw_dyn_vec)[:, None]
        )
        return f_gyr.reshape(-1)

    def _h_u(self, t, q, u):
        unodes = u.reshape(self.nnodes, -1)
        B_Omega = self.N_dyn @ unodes[:, 3:]

        # spin
        B_I_rho0 = self.cross_section_inertias.B_I_rho0
        B_L = B_Omega @ B_I_rho0.T

        f_gyr_qp_ubar = np.zeros((self.nquadrature_dyn_total, 6, 6))
        # B_Omega_tilde @ B_I_rho0 - B_L_tilde
        f_gyr_qp_ubar[:, 3:, 3:] = np.cross(
            B_I_rho0[None, :, :], B_Omega[:, :, None], axisa=1, axisb=1, axisc=1
        ) + ax2skew(B_L)

        return self.M_h_u_SAB.add_blocks(np.array([f_gyr_qp_ubar]))

    ###########################
    # unit-quaternion condition
    ###########################
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

    ####################################################
    # interactions with other bodies and the environment
    ####################################################
    def elDOF_P(self, xi):
        el = self.element_number(xi)
        start = (self.nnodes_element - 1) * el
        end = (self.nnodes_element - 1) * (el + 1) + 1
        return np.arange(7 * start, 7 * end)

    def elDOF_P_u(self, xi):
        el = self.element_number(xi)
        start = (self.nnodes_element - 1) * el
        end = (self.nnodes_element - 1) * (el + 1) + 1
        return np.arange(6 * start, 6 * end)

    def get_interaction_point(self, xi):
        # TODO: check that it is always done using this function, never access interaction points directly!
        if not (xi in self.interaction_points.keys()):
            if (node_number := self.node_number(xi)) is not False:
                nnodes = 1
                qDOF = np.arange(7) + 7 * node_number
                uDOF = np.arange(6) + 6 * node_number

                Nq = np.eye(7)
                Nu = np.eye(6)
                N = np.array([1.0])
            else:
                el = self.element_number(xi)
                nnodes = self.nnodes_element

                N = self.N_element(xi, el)
                Nq = np.zeros((7, 7 * nnodes))
                rows = np.arange(7)[:, None]
                cols = rows + 7 * np.arange(nnodes)
                Nq[rows, cols] = N

                Nu = np.zeros((6, 6 * nnodes))
                rows = np.arange(6)[:, None]
                cols = rows + 6 * np.arange(nnodes)
                Nu[rows, cols] = N

                qDOF = self.elDOF_P(xi)
                uDOF = self.elDOF_P_u(xi)

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
                zero_3_nqi=np.zeros((3, 7 * nnodes), dtype=float),
                zero_3_nui=np.zeros((3, 6 * nnodes), dtype=float),
                zero_3_nui_nqi=np.zeros((3, 6 * nnodes, 7 * nnodes), dtype=float),
            )
        return self.interaction_points[xi]

    def local_qDOF_P(self, xi):
        return self.get_interaction_point(xi).get("qDOF")

    def local_uDOF_P(self, xi):
        return self.get_interaction_point(xi).get("uDOF")

    ##########################
    # r_OP / A_IB contribution
    ##########################
    # TODO: move up to interactions?
    def node_number(self, xi):
        """For given xi in I = [0.0, 1.0], returns node number if xi is a node, otherwise False"""
        idx = np.where(self.xis_nodes == xi)[0]
        if len(idx) == 1:
            return idx[0]
        else:
            return False

    def element_number(self, xi):
        return np.where(self.xis_element_boundaries[:-1] <= xi)[0][-1]

    # cardillo functions
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

    # TODO: cache and move to init or upwards!
    def _A_IB(self, P):
        return Exp_SO3_quat(P)

    # TODO: cache
    def _A_IB_P(self, P):
        return Exp_SO3_quat_P(P)

    def _T_IB(self, P):
        # TODO: update this function in rotations
        return T_SO3_quat(P)

    def _T_IB_P(self, P):
        # TODO: update this function in rotations
        return T_SO3_quat_P(P)

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
    # internal virtual work #
    #########################
    # TODO: we need the mesh
    # TODO: can we use the same function for
    #   W_g and W_c
    #   g   and c
    # and use the same mesh?

    # Idea: make a W_compliance function, that is W_c for no constraints and W_g for fully constraint with caching and then
    #   W_g = W_compliance[:, gDOF]
    #   W_c = W_compliance[:, cDOF]
    # and similarly for g and c: l_compliance
    #   g = l_compliance[gDOF]
    #   c = l_compliance[cDOF] (+ Kc @ la_c)

    ########################
    # vectorized functions #
    ########################
    def l_sigma(self, q, u):
        _, B_gamma_bar, B_kappa_bar = self._eval_internal_vec(
            self.N_int, self.N_xi_int, q
        )

        epsilon_bar = np.empty((self.nquadrature_int_total, 6))
        epsilon_bar[:, :3] = B_gamma_bar - self.B_gamma0_bar
        epsilon_bar[:, 3:] = B_kappa_bar - self.B_kappa0_bar

        l = self.Nc_int.T @ (epsilon_bar * self.qw_int_vec[:, None])

        l_c = l[:, self.idx_c].reshape(-1)
        l_g = l[:, self.idx_g].reshape(-1)
        return l_c, l_g

    def l_sigma_q(self, q, u):
        # compute W_sigma
        A_IB, T, B_gamma_bar_P, B_kappa_bar_P = self._eval_internal_vec(
            self.N_int, self.N_xi_int, q, deval=True
        )

        # TODO: make sparse?

        # c_sigma_q_qp[qpi, N/N_xi, la_cDOF, qDOF]
        c_sigma_q_qp = np.empty((2, self.nquadrature_int_total, 6, 7))
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

    def _c_la_c_coo(self):
        C_qp = self.material_model.C_inv[self.idx_c[:, None], self.idx_c]
        c_la_c = self.c_la_c_SAB.add_blocks(
            np.array([[C_qp] * self.nquadrature_int_total])
        )

        self.__cla_c = c_la_c
        self.__cla_c_inv = spsolve(c_la_c.tocsc(), eye_array(self.nla_c, format="csc"))
        return c_la_c

    def W_sigma(self, q):
        # compute W_sigma
        A_IB, B_gamma_bar, B_kappa_bar = self._eval_internal_vec(
            self.N_int, self.N_xi_int, q
        )

        gamma_bar_tilde = ax2skew(B_gamma_bar)
        kappa_bar_tilde = ax2skew(B_kappa_bar)

        # W_sigma_qp[qpi, N/N_xi, uDOF, la_cDOF]
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
        sigma_qp = self.Nc_int @ la_sigma_nodes

        # compute Wla_sigma
        A_IB, B_gamma_bar, B_kappa_bar = self._eval_internal_vec(
            self.N_int, self.N_xi_int, q
        )

        # Wla_sigma_qp[qpi, N/N_xi, uDOF]
        Wla_sigma_qp = np.empty((2, self.nquadrature_int_total, 6))
        # to be multiplied with N_xi
        Wla_sigma_qp[1, :, :3] = -np.einsum("ijk,ik->ij", A_IB, sigma_qp[:, :3])
        Wla_sigma_qp[1, :, 3:] = -sigma_qp[:, 3:]

        # to be multiplied with N
        Wla_sigma_qp[0, :, :3] = 0.0
        Wla_sigma_qp[0, :, 3:] = np.cross(B_gamma_bar, sigma_qp[:, :3]) + np.cross(
            B_kappa_bar, sigma_qp[:, 3:]
        )

        # add together and multiply with quadrature weights
        Wla_sigma_nodes = self.N_int.T @ (
            Wla_sigma_qp[0] * self.qw_int_vec[:, None]
        ) + self.N_xi_int.T @ (Wla_sigma_qp[1] * self.qw_int_vec[:, None])
        return Wla_sigma_nodes.reshape(-1)

    def Wla_sigma_q(self, q, la_c=None, la_g=None):
        la_sigma_nodes = np.zeros((self.nnodes_sigma, 6))
        if la_c is not None:
            la_sigma_nodes[:, self.idx_c] = la_c.reshape(self.nnodes_sigma, -1)
        if la_g is not None:
            la_sigma_nodes[:, self.idx_g] = la_g.reshape(self.nnodes_sigma, -1)
        sigma_qp = self.Nc_int @ la_sigma_nodes

        # compute W_sigma
        A_IB, T, B_gamma_bar_P, B_kappa_bar_P = self._eval_internal_vec(
            self.N_int, self.N_xi_int, q, deval=True
        )

        I_n_P = np.einsum(
            "ijk,ikl->ijl",
            A_IB,
            np.cross(sigma_qp[:, :3, None], T, axisa=1, axisb=1, axisc=1),
        )

        # TODO: make sparse?
        # W_sigma_qp[qpi, N/N_xi, uDOF, qDOF]
        Wla_sigma_qp_qbar = np.zeros((3, self.nquadrature_int_total, 6, 7))
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

        return self.Wla_sigma_q_SAB.add_blocks(Wla_sigma_qp_qbar)

    ########################
    # evaluation functions #
    ########################
    def _eval_stresses(self, t, q, la_c, la_g, num_per_element): ...

    def _eval_straints(self, t, q, la_c, la_g, num_per_element): ...


def make_BoostedCosseratRod(
    *,
    polynomial_degree=None,
    idx_constraints=None,
    idx_displacement_based=None,
    nquadrature_int=None,
    nquadrature_dyn=None,
):
    # check if constraint indices are valid
    if idx_constraints is not None:
        if not (
            (np.array(idx_constraints) >= 0).all()
            & (np.array(idx_constraints) <= 5).all()
        ):
            raise ValueError("constraint values must between 0 and 5")
    else:
        idx_constraints = np.array([], dtype=int)
    idx_constraints = np.sort(idx_constraints)

    # check if displacement based indices are valid
    if idx_displacement_based is not None:
        if not (
            (np.array(idx_displacement_based) >= 0).all()
            & (np.array(idx_displacement_based) <= 5).all()
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

    polynomial_degree = 2 if polynomial_degree is None else polynomial_degree

    nquadrature_int = polynomial_degree if nquadrature_int == None else nquadrature_int
    # TODO: take trapezoidal rule as default?
    nquadrature_dyn = (
        int(np.ceil(3 / 2 * polynomial_degree + 1 / 2))
        if nquadrature_dyn == None
        else nquadrature_dyn
    )

    print(f"{nquadrature_int = }")
    print(f"{nquadrature_dyn = }")

    class BoostedCosseratRod_PetrovGalerkin(CosseratRod_PetrovGalerkin):
        def __init__(
            self,
            cross_section,
            material_model,
            nelement,
            *,
            Q,
            q0=None,
            u0=None,
            cross_section_inertias=CrossSectionInertias(),
            name="Cosserat_rod",
        ):
            """Petrov-Galerkin Cosserat rod formulations with
            quaternions for the nodal orientation parametrization.

            Parameters
            ----------
            cross_section : CrossSection
                Geometric cross-section properties: area, first and second moments
                of area.
            material_model: RodMaterialModel
                Constitutive law of Cosserat rod which relates the rod strain
                measures B_Gamma and B_Kappa with the contact forces B_n and couples
                B_m in the cross-section-fixed B-basis.
            nelement : int
                Number of rod elements.
            Q : np.ndarray (self.nq,)
                Generalized position coordinates of rod in a stress-free reference
                state. Q is a collection of nodal generalized position coordinates,
                which are given by the Cartesian coordinates of the nodal centerline
                point r_OP_i in R^3 together with non-unit quaternions p_i in R^4
                representing the nodal cross-section orientation.
            q0 : np.ndarray (self.nq,)
                Initial generalized position coordinates of rod at time t0.
            u0 : np.ndarray (self.nu,)
                Initial generalized velocity coordinates of rod at time t0.
                Generalized velocity coordinates u0 is a collection of the nodal
                generalized velocity coordinates, which are given by the nodal
                centerline velocity v_P_i in R^3 together with the cross-section
                angular velocity represented in the cross-section-fixed B-basis
                B_omega_IB.
            cross_section_inertias : CrossSectionInertias
                Inertia properties of cross-sections: Cross-section mass density and
                Cross-section inertia tensor represented in the cross-section-fixed
                B-Basis.
            name : str
                Name of contribution.
            """

            super().__init__(
                cross_section,
                material_model,
                cross_section_inertias,
                idx_constraints,
                idx_displacement_based,
                nelement,
                polynomial_degree,
                nquadrature_int,
                nquadrature_dyn,
                Q,
                q0,
                u0,
                name,
            )

        @staticmethod
        def straight_configuration(
            nelement,
            L,
            r_OP0=np.zeros(3, dtype=float),
            A_IB0=np.eye(3, dtype=float),
        ):
            """Compute generalized position coordinates for straight configuration."""
            nnodes = polynomial_degree * nelement + 1

            x0 = np.linspace(0, L, num=nnodes)
            y0 = np.zeros(nnodes)
            z0 = np.zeros(nnodes)
            r_OP = np.vstack((x0, y0, z0))
            p = Log_SO3_quat(A_IB0)
            rP = np.zeros((nnodes, 7), dtype=float)
            for i in range(nnodes):
                rP[i, :3] = r_OP0 + A_IB0 @ r_OP[:, i]
                rP[i, 3:] = p

            return rP.reshape(-1)

        # TODO: also copy&paste the other configurations
        # TODO: change order
        # The order is the same!

    return BoostedCosseratRod_PetrovGalerkin
