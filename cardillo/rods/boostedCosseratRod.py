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
            include_f_gyr = False
        else:
            self.cross_section_inertias = cross_section_inertias
            include_f_gyr = True

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

        # element intervals
        self.element_interval = mesh_kin.element_interval
        self.element_number = mesh_kin.element_number
        self.node_number = mesh_kin.node_number

        # total number of nodes and per element
        self.nnodes = mesh_kin.nnodes
        self.nnodes_element = mesh_kin.nnodes_element

        # total number of generalized position and velocity coordinates
        self.nq = self.nnodes * 7
        self.nu = self.nnodes * 6

        self.N = lambda xis, els: mesh_kin.shape_functions(xis, els, 1)
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

        self.Nc = lambda xis, els: mesh_cg.shape_functions(xis, els, 0)[0]
        self.Nc_element = lambda xi, el: mesh_cg.shape_function_array_element(xi, el, 0)

        quadrature_int_cg = mesh_cg.quadrature(nquadrature_int, "Gauss", 0)
        self.Nc_int = quadrature_int_cg["N"][0]

        # total number of compliance coordinates
        self.nla_sigma = self.nnodes_sigma * 6
        if (nla_c := self.nnodes_sigma * len(self.idx_c)) > 0:
            self.nla_c = self._nla_c = nla_c

            ##############
            # compliance #
            # c = c_la_c @ la_c - l_c
            ##############
            # la_c = -c_la_c @ c(q, u, 0)
            self.la_c = lambda t, q, u: self.__cla_c_inv @ self.l_sigma(q)[0]
            self.c = lambda t, q, u, la_c: self.__cla_c @ la_c - self.l_sigma(q)[0]
            self.c_q = lambda t, q, u, la_c: -self.l_sigma_q(q)[0]
            self.c_la_c = lambda: self.__cla_c
            self.W_c = lambda t, q: self.W_sigma(q)[0]
            self.Wla_c_q = lambda t, q, la_c: self.Wla_sigma_q(q, la_c, None)
            self.E_pot_comp = self._E_pot_comp
        else:
            self._nla_c = 0

        if (nla_g := self.nnodes_sigma * len(self.idx_g)) > 0:
            self.nla_g = self._nla_g = nla_g

            #############
            # constraints
            #  g = - l_g
            #############
            self.nu_zeros = np.zeros(self.nu)
            self.g = lambda t, q: -self.l_sigma(q)[1]
            self.g_q = lambda t, q: -self.l_sigma_q(q)[1]
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
            self._nDB = len(self.idx_db)
            include_f_pot = True
            self.E_pot = self._E_pot
        else:
            self._nDB = 0
            include_f_pot = False

        # compose h vector
        if include_f_gyr:
            if include_f_pot:
                self.h = lambda t, q, u: self.f_gyr(u) + self.f_pot(q)
                self.h_q = lambda t, q, u: self.f_pot_q(q)
                self.h_u = lambda t, q, u: self.f_gyr_u(u)
            else:
                self.h = lambda t, q, u: self.f_gyr(u)
                self.h_u = lambda t, q, u: self.f_gyr_u(u)
        else:
            if include_f_pot:
                self.h = lambda t, q, u: self.f_pot(q)
                self.h_q = lambda t, q, u: self.f_pot_q(q)

        # reference strains for weight of blocks
        self.set_reference_strains(Q)

        # TODO: do inner dict with enum as key or use a class instead of dict
        self.interaction_points: dict[float, dict[str, np.ndarray]] = {}

        ################
        # assemble matrices for block structure
        ################
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
            (6, 7),
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
        self.h_pot_q_SAB = SparseArrayBlocks((self.nu, self.nq), (6, 7), h_pot_q_pairs)

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

    def set_reference_strains(self, Q):
        self.Q = Q.copy()

        _, B_gamma0_bar, B_kappa0_bar = self._eval_internal_vec(
            self.N_int, self.N_xi_int, self.Q
        )
        self.J_int_vec = np.linalg.norm(B_gamma0_bar, axis=1)
        self.B_gamma0_bar_int = B_gamma0_bar
        self.B_kappa0_bar_int = B_kappa0_bar
        self.epsilon0_int = (
            np.hstack([B_gamma0_bar, B_kappa0_bar]) / self.J_int_vec[:, None]
        )

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

    ############################
    # total energies and momenta
    ############################
    # the potential energies only work if
    # A) there is no coupling between the deformations
    # B) all of the coupled deformations are (not) db
    #    --> we put a warning when there are db and mx deformations
    def _E_pot_comp(self, t, q, la_c):
        if self._nDB > 0:
            warn(
                "E_pot_comp might not be correct if there are displacement based deformations."
            )
        _eval = self._eval_internal_vec(self.N_int, self.N_xi_int, q)
        epsilon_db = np.hstack([_eval[1], _eval[2]]) / self.J_int_vec[:, None]

        la_sigma_nodes = np.zeros((self.nnodes_sigma, 6))
        la_sigma_nodes[:, self.idx_c] = la_c.reshape(self.nnodes_sigma, -1)
        la_sigma = self.Nc_int @ la_sigma_nodes

        comp_pot = self.material_model.potential_comp_vec(la_sigma)

        E_pot_i = (
            np.einsum("ij,ij->i", epsilon_db - self.epsilon0_int, la_sigma) - comp_pot
        )
        E_pot = np.sum(E_pot_i * self.qw_int_vec * self.J_int_vec)
        return E_pot

    def _E_pot(self, t, q):
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
        E_pot_i = self.material_model.potential_vec(epsilon, epsilon0)
        E_pot = np.sum(E_pot_i * self.qw_int_vec * self.J_int_vec)
        return E_pot

    def E_kin(self, t, q, u):
        unodes = u.reshape(self.nnodes, -1)
        vO = self.N_dyn @ unodes

        v = vO[:, :3]
        B_Omega = vO[:, 3:]

        A_rho0 = self.cross_section_inertias.A_rho0
        B_I_rho0 = self.cross_section_inertias.B_I_rho0
        E_kin_i = 0.5 * (
            A_rho0 * np.sum(v * v, axis=1)
            + np.einsum("ij,jk,ik->i", B_Omega, B_I_rho0, B_Omega)
        )

        E_kin = np.sum(E_kin_i * self.qw_dyn_vec * self.J_dyn_vec)
        return E_kin

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

    def f_gyr(self, u):
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

    def f_gyr_u(self, u):
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
        Wla_sigma_qp_qbar = np.zeros((4, self.nquadrature_int_total, 6, 7))
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

    def f_pot(self, q):
        _eval = self._eval_internal_vec(self.N_int, self.N_xi_int, q)
        epsilon = np.hstack([_eval[1], _eval[2]]) / self.J_int_vec[:, None]
        sigma_db = self.material_model.sigma(epsilon, self.epsilon0_int)

        sigma_qp = np.zeros((self.nquadrature_int_total, 6))
        sigma_qp[:, self.idx_db] = sigma_db[:, self.idx_db]

        return self.h_pot(_eval, sigma_qp)

    def f_pot_q(self, q):
        _eval, _deval = self._eval_internal_vec(
            self.N_int, self.N_xi_int, q, deval=True
        )
        epsilon = np.hstack([_eval[1], _eval[2]]) / self.J_int_vec[:, None]
        sigma_db = self.material_model.sigma(epsilon, self.epsilon0_int)

        sigma_qp = np.zeros((self.nquadrature_int_total, 6))
        sigma_qp[:, self.idx_db] = sigma_db[:, self.idx_db]

        A_IB, B_gamma_bar, B_kappa_bar = _eval
        T, B_gamma_bar_P, B_kappa_bar_P = _deval

        ######################
        # material stiffness #
        ######################
        A_IB_transpose_to_J = A_IB.transpose(0, 2, 1) / self.J_int_vec[:, None, None]
        T_to_J = T / self.J_int_vec[:, None, None]

        sigma_epsilon = self.material_model.sigma_epsilon(epsilon, self.epsilon0_int)
        B_n_gamma, B_n_kappa, B_m_gamma, B_m_kappa = sigma_epsilon

        # fmt: off
        B_n_P = (B_n_gamma @ B_gamma_bar_P + B_n_kappa @ B_kappa_bar_P) / self.J_int_vec[:, None, None]
        B_m_P = (B_m_gamma @ B_gamma_bar_P + B_m_kappa @ B_kappa_bar_P) / self.J_int_vec[:, None, None]
        # fmt: on

        # TODO: make sparse?
        # f_pot_qp_qbar[N/N_xi, qpi, uDOF, qDOF]
        f_pot_qp_qbar = np.zeros((4, self.nquadrature_int_total, 6, 7))
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
                xi0, xi1 = self.element_interval[el]
                xis.append(np.linspace(xi0, xi1, n_per_element))
                els.append(np.tile(el, n_per_element))
            xis = np.concatenate(xis)
            els = np.concatenate(els)

        return xis, els

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
            sigma_db = self.material_model.sigma(epsilon, epsilon0)
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

            C_inv = self.material_model.C_inv[self.idx_c[:, None], self.idx_c]
            # TODO: position dependent material law
            epsilon[:, self.idx_c] = la_sigma @ C_inv.T

        # strains from constraints are always 0
        epsilon[:, self.idx_g] = 0.0

        return xis, epsilon[:, :3], epsilon[:, 3:]


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
        idx_constraints = np.asarray(idx_constraints, dtype=int)
        if not ((idx_constraints >= 0).all() & (idx_constraints <= 5).all()):
            raise ValueError("constraint values must between 0 and 5")
    else:
        idx_constraints = np.array([], dtype=int)
    idx_constraints = np.sort(idx_constraints)

    # check if displacement based indices are valid
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
                measures B_gamma and B_kappa with the contact forces B_n and couples
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
            r_OP0=zeros3,
            A_IB0=eye3,
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
            r0 = np.zeros((nnodes, 3))
            p0 = np.zeros((nnodes, 4))

            for i, xii in enumerate(xis):
                r0[i] = r_OP0 + A_IB0 @ r_OP(xii)
                A_IBi = A_IB0 @ A_IB(xii)
                p0[i] = Log_SO3_quat(A_IBi)

            # check for the right quaternion hemisphere
            for i in range(nnodes - 1):
                inner = p0[i] @ p0[i + 1]
                if inner < 0:
                    p0[i + 1] *= -1

            return np.hstack([r0, p0]).reshape(-1)

        # TODO: also copy&paste the other configurations
        # TODO: change order
        # The order is the same!

    return BoostedCosseratRod_PetrovGalerkin
