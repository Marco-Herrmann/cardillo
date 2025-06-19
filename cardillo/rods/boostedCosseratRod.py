import numpy as np
from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey
from warnings import warn


from abc import ABC, abstractmethod
from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey
import numpy as np
from scipy.sparse.linalg import spsolve

from cardillo.math.algebra import norm, cross3, ax2skew
from cardillo.math.approx_fprime import approx_fprime
from cardillo.math.rotations import Exp_SO3_quat, T_SO3_inv_quat, T_SO3_inv_quat_P
from cardillo.utility.coo_matrix import CooMatrix

from ._base_export import RodExportBase
from ._cross_section import CrossSectionInertias
from .discretization.lagrange import LagrangeKnotVector
from .discretization.mesh1D import Mesh1D

eye3 = np.eye(3, dtype=float)
eye4 = np.eye(4, dtype=float)


from cardillo.math import (
    norm,
    cross3,
    T_SO3_quat,
    T_SO3_quat_P,
    Exp_SO3_quat,
    Exp_SO3_quat_p,
    Log_SO3_quat,
)
from cardillo.utility.check_time_derivatives import check_time_derivatives

from ._base_export import RodExportBase

from ._base import (
    CosseratRodDisplacementBased,
    CosseratRodMixed,
    make_CosseratRodConstrained,
)
from ._cross_section import CrossSectionInertias


class CosseratRod_PetrovGalerkin(RodExportBase):
    def __init__(
        self,
        cross_section,
        material_model,
        cross_section_inertias,
        constraints,
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
        self.cross_section_inertias = cross_section_inertias
        self.idx_constrained = np.atleast_1d(constraints)
        self.idx_impressed = np.setdiff1d(np.arange(6), self.idx_constrained)

        self.n_constrained = len(self.idx_constrained)
        self.n_impressed = len(self.idx_impressed)

        self.name = "Cosserat_rod" if name is None else name

        self.nelement = nelement
        self.polynomial_degree = polynomial_degree

        self.nquadrature_int = nquadrature_int
        self.nquadrature_dyn = nquadrature_dyn

        self.knot_vector = LagrangeKnotVector(polynomial_degree, nelement)
        self.xis_nodes = self.knot_vector.data
        self.xis_element_boundaries = self.knot_vector.data[:: polynomial_degree - 1]

        # mesh for interpolation of (r, P) and (v, omega)
        mesh_kin = Mesh1D(
            knot_vector=self.knot_vector,
            nquadrature=1,  # dummy value
            dim_q=7,
            derivative_order=1,
            basis="Lagrange",
            dim_u=6,
        )

        # total number of nodes and per element
        self.nnodes = mesh_kin.nnodes
        self.nnodes_element = mesh_kin.nnodes_per_element

        # total number of generalized position and velocity coordinates
        self.nq = mesh_kin.nq
        self.nu = mesh_kin.nu

        self.nq_r = int(self.nq / 7 * 3)
        self.nu_r = int(self.nu / 6 * 3)

        # number of generalized position and velocity coordinates per element
        self.nq_element = mesh_kin.nq_per_element
        self.nu_element = mesh_kin.nu_per_element

        self.nq_element_r = int(self.nq_element / 7 * 3)
        self.nu_element_r = int(self.nu_element / 6 * 3)

        # global element connectivity
        # qe = q[elDOF[e]] "q^e = C_q,e q"
        self.elDOF = mesh_kin.elDOF
        # ue = u[elDOF_u[e]] "u^e = C_u,e u"
        self.elDOF_u = mesh_kin.elDOF_u

        # global nodal connectivity
        # position
        self.nodalDOF = mesh_kin.nodalDOF
        self.nodalDOF_r = self.nodalDOF[:, :3]
        self.nodalDOF_p = self.nodalDOF[:, 3:]
        # velocity
        self.nodalDOF_u = mesh_kin.nodalDOF_u
        self.nodalDOF_u_v = self.nodalDOF_u[:, :3]
        self.nodalDOF_u_o = self.nodalDOF_u[:, 3:]

        # nodal connectivity on element level
        # (r_OP, P)_i^e = C_i^e * C_q,e q = C_i^e * q^e
        # (r_OP, P)_i = qe[nodalDOF_element_r[i]]
        self.nodalDOF_element = mesh_kin.nodalDOF_element
        self.nodalDOF_element_r = self.nodalDOF_element[:, :3]
        self.nodalDOF_element_p = self.nodalDOF_element[:, 3:]
        # (v_P, P)_i^e = Cu_i^e * C_u,e u = Cu_r,i^e * u^e
        # (v_P, P)_i = ue[nodalDOF_element_r[i]]
        self.nodalDOF_element_u = mesh_kin.nodalDOF_element_u
        self.nodalDOF_element_u_v = self.nodalDOF_element_u[:, :3]
        self.nodalDOF_element_u_o = self.nodalDOF_element_u[:, 3:]

        # evaluate shape functions at specific xi
        self.basis_functions = mesh_kin.eval_basis
        self.N_q = mesh_kin.eval_basis_matrix_q
        self.N_u = mesh_kin.eval_basis_matrix_u

        #####################
        # quadrature points #
        #####################
        self.qp_int, self.qw_int = mesh_kin.quadrature_points(self.nquadrature_int)
        self.qp_dyn, self.qw_dyn = mesh_kin.quadrature_points(self.nquadrature_dyn)

        #####
        # TODO: get them in the most performant way
        #####
        # shape functions and their first derivatives
        # for quadrature points of internal virtual work
        N_mtx_q, N_mtx_u = mesh_kin.shape_functions_matrix(self.nquadrature_int)
        self.N_int_q, self.N_int_q_xi = N_mtx_q
        self.N_int_u, self.N_int_u_xi = N_mtx_u

        self.N_int_omega = self.N_int_u[:, :, 3:, self.nu_element_r :]

        # for quadrature points of dynamic virtual work
        N_mtx_q, N_mtx_u = mesh_kin.shape_functions_matrix(self.nquadrature_dyn)
        self.N_dyn_q, self.N_dyn_q_xi = N_mtx_q
        self.N_dyn_u, self.N_dyn_u_xi = N_mtx_u

        self.N_dyn_omega = self.N_dyn_u[:, :, 3:, self.nu_element_r :]

        # referential generalized position coordinates, initial generalized
        # position and velocity coordinates
        self.Q = Q
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else u0

        # unit quaternion constraints
        dim_g_S = 1
        self.nla_S = self.nnodes * dim_g_S
        self.nodalDOF_la_S = np.arange(self.nla_S).reshape(self.nnodes, dim_g_S)
        self.la_S0 = np.zeros(self.nla_S, dtype=float)

        ###############################
        # compliance and constrtaints #
        ###############################
        self.knot_vector_sigma = LagrangeKnotVector(polynomial_degree - 1, nelement)
        self.xis_nodes_sigma = self.knot_vector_sigma.data

        # mesh for interpolation of (B_n, B_m) Resultant Contact Stresses
        mesh_rcs = Mesh1D(
            knot_vector=self.knot_vector_sigma,
            nquadrature=1,  # dummy value
            dim_q=6,
            derivative_order=0,
            basis="Lagrange_Disc",
            dim_u=6,
        )

        self.nnodes_compliance = mesh_rcs.nnodes_per_element

        # total number of compliance coordinates
        self.nla_compliance = mesh_rcs.nq
        self.nla_c = int(mesh_rcs.nq / 6 * self.n_impressed)
        self.nla_g = int(mesh_rcs.nq / 6 * self.n_constrained)

        # TODO: check what actually is used from here and prepare for hard constraint

        # total number of nodes
        self.nnodes_la_c = mesh_rcs.nnodes

        # number of nodes per element
        self.nnodes_element_la_c = mesh_rcs.nnodes_per_element

        # total number of compliance coordinates
        self.nla_c = mesh_rcs.nq

        # number of compliance coordinates per element
        self.nla_c_element = mesh_rcs.nq_per_element

        # global element connectivity for compliance coordinates
        self.elDOF_la_c = mesh_rcs.elDOF

        # global nodal connectivity
        self.nodalDOF_la_c = mesh_rcs.nodalDOF

        # nodal connectivity on element level
        self.nodalDOF_element_la_c = mesh_rcs.nodalDOF_element

        # shape functions and their first derivatives
        self.N_la_c = mesh_rcs.N

        # evaluate shape functions at specific xi
        self.basis_functions_la_c = mesh_rcs.eval_basis

        self.N_int_c, _ = mesh_rcs.shape_functions_matrix(self.nquadrature_int)

        # TODO: this splits c and g from l_c_el
        self.cDOF_el = np.arange(self.nla_c_element)

        self.set_reference_strains(self.Q)

        # caches
        self._cache_f_gyr = LRUCache(self.nquadrature_dyn * nelement)
        self._cache_internal = LRUCache(self.nquadrature_int * nelement)

    def set_reference_strains(self, Q):
        self.Q = Q.copy()

        # precompute values of the reference configuration in order to save
        # computation time J in Harsch2020b (5)
        self.J_int = np.zeros((self.nelement, self.nquadrature_int), dtype=float)
        self.J_dyn = np.zeros((self.nelement, self.nquadrature_dyn), dtype=float)
        # strains of the reference configuration
        self.epsilon0 = np.zeros((self.nelement, self.nquadrature_int, 6), dtype=float)
        self.epsilon0_bar = np.zeros(
            (self.nelement, self.nquadrature_int, 6), dtype=float
        )

        for el in range(self.nelement):
            qe = self.Q[self.elDOF[el]]

            for i in range(self.nquadrature_int):
                # evaluate required quantities
                qpi = self.qp_int[el, i]
                N = self.N_int_q[el, i]
                N_xi = self.N_int_q_xi[el, i]
                epsilon_bar = self._eval(qpi, N, N_xi, qe)[0][2]

                # save length of reference tangential vector and strain
                self.J_int[el, i] = norm(epsilon_bar[:3])
                self.epsilon0[el, i] = epsilon_bar / self.J_int[el, i]
                self.epsilon0_bar[el, i] = epsilon_bar

            for i in range(self.nquadrature_dyn):
                # evaluate required quantities
                qpi = self.qp_dyn[el, i]
                N = self.N_dyn_q[el, i]
                N_xi = self.N_dyn_q_xi[el, i]
                epsilon_bar = self._eval(qpi, N, N_xi, qe)[0][2]

                # length of reference tangential vector
                self.J_dyn[el, i] = norm(epsilon_bar[:3])

    # TODO: When are these functions called? Do we need and can we speed them up?
    def element_number(self, xi):
        """Compute element number from given xi."""
        return self.knot_vector.element_number(xi)[0]

    def element_interval(self, el):
        return self.knot_vector.element_interval(el)

    ############################
    # export of centerline nodes
    ############################
    def nodes(self, q):
        """Returns nodal position coordinates"""
        q_body = q[self.qDOF]
        return np.array([q_body[nodalDOF] for nodalDOF in self.nodalDOF_r]).T

    def nodalFrames(self, q, elementwise=False):
        """Returns nodal positions and nodal directors.
        If elementwise==True : returned arrays are each of shape [nnodes, 3]
        If elementwise==False : returned arrays are each of shape [nelements, nnodes_per_element, 3]
        """

        q_body = q[self.qDOF]
        if elementwise:
            r = np.zeros([self.nelement, self.nnodes_element, 3], dtype=float)
            ex = np.zeros([self.nelement, self.nnodes_element, 3], dtype=float)
            ey = np.zeros([self.nelement, self.nnodes_element, 3], dtype=float)
            ez = np.zeros([self.nelement, self.nnodes_element, 3], dtype=float)
            for el, elDOF in enumerate(self.elDOF):
                qe = q_body[elDOF]
                rP = np.array(
                    [qe[nodalDOF_el] for nodalDOF_el in self.nodalDOF_element]
                )
                r[el] = rP[:, :3]
                A_IB = np.array(
                    [
                        Exp_SO3_quat(rP[i, 3:], normalize=True)
                        for i in range(self.nnodes_element)
                    ]
                )

                ex[el] = A_IB[:, :, 0]
                ey[el] = A_IB[:, :, 1]
                ez[el] = A_IB[:, :, 2]

            return r, ex, ey, ez

        else:
            rP = np.array([q_body[nodalDOF] for nodalDOF in self.nodalDOF])
            r = rP[:, :3]
            p = rP[:, 3:]
            A_IB = np.array(
                [Exp_SO3_quat(p[i], normalize=True) for i in range(self.nnodes)]
            )
            return r, A_IB[:, :, 0], A_IB[:, :, 1], A_IB[:, :, 2]

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
        q_dot = np.zeros_like(q, dtype=np.common_type(q, u))

        for node in range(self.nnodes):
            # centerline time derivative from centerline velocities
            q_dot[self.nodalDOF_r[node]] = u[self.nodalDOF_u_v[node]]

            # quaternion time derivative from angular velocities
            nodalDOF_p = self.nodalDOF_p[node]
            q_dot[nodalDOF_p] = (
                T_SO3_inv_quat(q[nodalDOF_p], normalize=True)
                @ u[self.nodalDOF_u_o[node]]
            )

        return q_dot

    def q_dot_q(self, t, q, u):
        return approx_fprime(q, lambda q_: self.q_dot(t, q_, u))

    def q_dot_u(self, t, q, u):
        return approx_fprime(u, lambda u_: self.q_dot(t, q, u_))

    def step_callback(self, t, q, u):
        """ "Quaternion normalization after each time step."""
        for node in range(self.nnodes):
            p_DOF = self.nodalDOF[node][3:]
            p = q[p_DOF]
            q[p_DOF] = p / norm(p)
        return q, u

    ##########################################
    # TODO: Do we need energies and momenta? #
    ##########################################

    ##################
    # eval functions #
    ##################
    # TODO: cache this
    def _eval_r_A(self, xi, qe, el, N):
        rP = N @ qe
        A_IB = Exp_SO3_quat(rP[3:])
        A_IB_qe = Exp_SO3_quat_p(rP[3:], normalize=True) @ N[3:]
        return (rP[:3], A_IB), (N[:3], A_IB_qe)

    # TODO: cache this, not sure if we need to ...
    def _eval(self, xi, N, N_xi, qe):
        # eval
        rP = N @ qe
        rP_xi = N_xi @ qe
        A_IB = Exp_SO3_quat(rP[3:])
        T = T_SO3_quat(rP[3:], normalize=True)
        B_gamma = A_IB.T @ rP_xi[:3]
        B_kappa = T @ rP_xi[3:]
        eps = np.array([*B_gamma, *B_kappa])

        # deval
        A_IB_qe = Exp_SO3_quat_p(rP[3:], normalize=True) @ N[3:]
        # TODO: think of implementing (T_SO3_quat(P) @ Q)_P for kappa
        # (ca. 10% speed up per call)
        TP_xi_P = rP_xi[3:] @ T_SO3_quat_P(rP[3:], normalize=True)
        B_gamma_bar_qe = ax2skew(B_gamma) @ T @ N[3:] + A_IB.T @ N_xi[:3]
        B_kappa_bar_qe = TP_xi_P @ N[3:] + T @ N_xi[3:]
        eps_qe = np.vstack([B_gamma_bar_qe, B_kappa_bar_qe])
        return (rP[:3], A_IB, eps), (N[:3], A_IB_qe, eps_qe)

    #########################################
    # equations of motion
    #########################################
    def M(self, t, q):
        return self.__M

    def _M_coo(self):
        """ "Mass matrix is called in assembler callback."""
        self.__M = CooMatrix((self.nu, self.nu))
        for el in range(self.nelement):
            # extract element degrees of freedom
            elDOF_u = self.elDOF_u[el]

            # sparse assemble element mass matrix
            self.__M[elDOF_u, elDOF_u] = self.M_el(el)

    def M_el(self, el):
        M_el = np.zeros((self.nu_element, self.nu_element), dtype=float)

        for i in range(self.nquadrature_dyn):
            # extract reference state variables
            Mi = (
                self.cross_section_inertias.generalized_inertia
                * self.qw_dyn[el, i]
                * self.J_dyn[el, i]
            )
            N_dyn = self.N_dyn_u[el, i]
            M_el += N_dyn.T @ Mi @ N_dyn

        return M_el

    def h(self, t, q, u):
        h = np.zeros(self.nu, dtype=u.dtype)
        for el in range(self.nelement):
            elDOF_u = self.elDOF_u[el][self.nu_element_r :]
            h[elDOF_u] -= self.f_gyr_master(u[elDOF_u], el)[0]
        return h

    def h_u(self, t, q, u):
        coo = CooMatrix((self.nu, self.nu))
        for el in range(self.nelement):
            elDOF_u = self.elDOF_u[el][self.nu_element_r :]
            coo[elDOF_u, elDOF_u] = -self.f_gyr_master(u[elDOF_u], el)[1]
        return coo

    @cachedmethod(
        lambda self: self._cache_f_gyr,
        key=lambda self, ue_red, el: hashkey(*ue_red, el),
    )
    def f_gyr_master(self, ue_red, el):
        # only compute part with omega
        dim_red = self.nu_element - self.nu_element_r
        f_gyr_el = np.zeros(dim_red, dtype=ue_red.dtype)
        f_gyr_el_ue = np.zeros((dim_red, dim_red), dtype=ue_red.dtype)

        for i in range(self.nquadrature_dyn):
            # interpoalte angular velocity
            N_dyn = self.N_dyn_omega[el, i]
            B_Omega = N_dyn @ ue_red
            weight = self.J_dyn[el, i] * self.qw_dyn[el, i]

            # vector of gyroscopic forces
            B_I_rho0 = self.cross_section_inertias.B_I_rho0
            f_gyr_el_p = cross3(B_Omega, B_I_rho0 @ B_Omega) * weight
            f_gyr_u_el_p = (
                (ax2skew(B_Omega) @ B_I_rho0 - ax2skew(B_I_rho0 @ B_Omega))
            ) * weight

            # multiply vector of gyroscopic forces with nodal virtual rotations
            f_gyr_el += N_dyn.T @ f_gyr_el_p
            f_gyr_el_ue += N_dyn.T @ f_gyr_u_el_p @ N_dyn

        return f_gyr_el, f_gyr_el_ue

    ###########################
    # unit-quaternion condition
    ###########################
    def g_S(self, t, q):
        # TODO: Can this be optimized?
        P = q[self.nq_r :].reshape(4, -1)
        return np.sum(P**2, axis=0) - 1

    def g_S_q(self, t, q):
        # TODO: Can this be optimized?
        coo = CooMatrix((self.nla_S, self.nq))
        coo.data = 2 * q[self.nq_r :]
        coo.row = np.tile(np.arange(self.nla_S), 4)
        coo.col = np.arange(self.nq_r, self.nq)
        return coo

    ####################################################
    # interactions with other bodies and the environment
    ####################################################
    def elDOF_P(self, xi):
        el = self.element_number(xi)
        return self.elDOF[el]

    def elDOF_P_u(self, xi):
        el = self.element_number(xi)
        return self.elDOF_u[el]

    def local_qDOF_P(self, xi):
        return self.elDOF_P(xi)

    def local_uDOF_P(self, xi):
        return self.elDOF_P_u(xi)

    ##########################
    # r_OP / A_IB contribution
    ##########################
    def node_number(self, xi):
        """For given xi in I = [0.0, 1.0], returns element node number if xi is a node, otherwise False"""
        idx = np.where(self.xis_nodes == xi)[0]
        if len(idx) == 1:
            return idx[0] % self.polynomial_degree
        else:
            return False

    def element_number(self, xi):
        return np.where(self.xis_element_boundaries >= xi)[0][0]

    def _r_A_nodal(self, qe, node):
        rP = qe[self.nodalDOF_element[node]]
        return rP[:3], Exp_SO3_quat(rP[3:], normalize=False)

    def _r_A_nodal_qe(self, qe, node):
        nodalDOF = self.nodalDOF_element[node]
        rP = qe[nodalDOF]
        r_OP_qe = np.zeros((3, self.nq_element), dtype=qe.dtype)
        A_IB_qe = np.zeros((3, 3, self.nq_element), dtype=qe.dtype)
        r_OP_qe[:, nodalDOF[:3]] = eye3
        A_IB_qe[:, :, nodalDOF[3:]] = Exp_SO3_quat_p(rP[3:], normalize=False)
        return r_OP_qe, A_IB_qe

    def _v_Om_nodal(self, ue, node):
        vo = ue[self.nodalDOF_element_u[node]]
        return vo[:3], vo[3:]

    # cardillo functions
    def r_OP(self, t, qe, xi, B_r_CP=None):
        # TODO: think of a good way to get B_r_CP incorporated here
        if node := self.node_number(xi):
            r_OC, A_IB = self._r_A_nodal(qe, node)
        else:
            N, N_xi = self.N_q(xi)
            r_OC, A_IB = self._eval_r_A(qe, xi, N, N_xi)[0]

        return r_OC if B_r_CP is None else r_OC + A_IB @ B_r_CP

    def r_OP_q(self, t, qe, xi, B_r_CP=None): ...
    def v_P(self, t, qe, ue, xi, B_r_CP=None):
        if node := self.node_number(xi):
            # TODO: we don't need A_IB always!
            _, A_IB = self._r_A_nodal(qe, node)
            v_C, B_Omega = self._v_Om_nodal(ue, node)
        else:
            pass

        return v_C if B_r_CP is None else v_C + A_IB @ cross3(B_Omega, B_r_CP)

    def v_P_q(self, t, qe, ue, xi, B_r_CP=None): ...
    def J_P(self, t, qe, xi, B_r_CP=None): ...
    def J_P_q(self, t, qe, xi, B_r_CP=None): ...
    def a_P(self, t, qe, ue, ue_dot, xi, B_r_CP=None):
        if node := self.node_number(xi):
            # TODO: we don't need A_IB and B_Omega always!
            _, A_IB = self._r_A_nodal(qe, node)
            _, B_Omega = self._v_Om_nodal(ue, node)
            a_C, B_Psi = self._v_Om_nodal(ue_dot, node)
        else:
            pass

        if B_r_CP is None:
            return a_C
        else:
            return a_C + A_IB @ (
                cross3(B_Psi, B_r_CP) + cross3(B_Omega, cross3(B_Omega, B_r_CP))
            )

    def a_P_q(self, t, qe, ue, ue_dot, xi, B_r_CP=None): ...
    def a_P_u(self, t, qe, ue, ue_dot, xi, B_r_CP=None): ...

    def A_IB(self, t, qe, xi):
        if node := self.node_number(xi):
            _, A_IB = self._r_A_nodal(qe, node)
        else:
            N, N_xi = self.N_q(xi)
            _, A_IB = self._eval_r_A(qe, xi, N, N_xi)[0]
        return A_IB

    def A_IB_q(self, t, qe, xi): ...
    def B_Omega(self, t, qe, ue, xi): ...
    def B_Omega_q(self, t, qe, ue, xi): ...
    def B_J_R(self, t, qe, xi): ...
    def B_J_R_q(self, t, qe, xi): ...
    def B_Psi(self, t, qe, ue, ue_dot, xi): ...
    def B_Psi_q(self, t, qe, ue, ue_dot, xi): ...
    def B_Psi_u(self, t, qe, ue, ue_dot, xi): ...

    ##############
    # compliance #
    ##############
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

    ####################################################
    # element functions for compliance and constraints #
    ####################################################
    @cachedmethod(
        lambda self: self._cache_internal,
        key=lambda self, qe, el: hashkey(*qe, el),
    )
    def internal_master(self, qe, el):
        c_el = np.zeros(self.nla_c_element, dtype=qe.dtype)
        c_el_qe = np.zeros((self.nla_c_element, self.nq_element), dtype=qe.dtype)
        W_c_el = np.zeros((self.nu_element, self.nla_c_element), dtype=qe.dtype)

        for i in range(self.nquadrature_int):
            # extract reference state variables
            qpi = self.qp_int[el, i]
            qwi = self.qw_int[el, i]
            epsilon0_bar = self.epsilon0_bar[el, i]

            # get shape functions
            N_q = self.N_int_q[el, i]
            N_q_xi = self.N_int_q_xi[el, i]
            N_omega = self.N_int_omega[el, i]
            N_u_xi = self.N_int_u_xi[el, i]
            N_compliance = self.N_int_c[el, i]

            # get stretches
            _eval, _deval = self._eval(qpi, N_q, N_q_xi, qe)
            epsilon_bar = _eval[2]
            epsilon_bar_qe = _deval[2]

            # constributions to W
            mtx1 = np.zeros((6, 6), dtype=float)
            mtx1[:3, :3] = _eval[1] * qwi
            mtx1[3:, 3:] = eye3 * qwi

            mtx2_omega = np.hstack(
                [ax2skew(epsilon_bar[:3] * qwi), ax2skew(epsilon_bar[3:] * qwi)]
            )

            # compose vector and matrices
            c_el += N_compliance.T @ ((epsilon0_bar - epsilon_bar) * qwi)
            c_el_qe -= N_compliance.T @ epsilon_bar_qe * qwi
            W_c_el -= N_u_xi.T @ mtx1 @ N_compliance
            W_c_el[self.nu_element_r :] += N_omega.T @ (mtx2_omega @ N_compliance)

        return c_el, c_el_qe, W_c_el

    def Wla_compliance_qe(self, qe, la, el): ...

    ##############
    # compliance #
    ##############
    # K_c^{-1}
    def c_la_c(self):
        return self.__c_la_c

    def _c_la_c_coo(self):
        self.__c_la_c = CooMatrix((self.nla_c, self.nla_c))
        self.K_c_inv = np.zeros(
            (self.nelement, self.nla_c_element, self.nla_c_element), dtype=float
        )
        for el in range(self.nelement):
            elDOF_la_c = self.elDOF_la_c[el]
            K_c_inv_el = self.c_la_c_el(el)
            self.__c_la_c[elDOF_la_c, elDOF_la_c] = K_c_inv_el
            self.K_c_inv[el] = K_c_inv_el

    def c_la_c_el(self, el):
        c_la_c_el = np.zeros((self.nla_c_element, self.nla_c_element))
        for i in range(self.nquadrature_int):
            Ci_inv = self.material_model.C_inv * self.qw_int[el, i] * self.J_int[el, i]
            N_c = self.N_int_c[el, i][self.idx_impressed[:, None], self.cDOF_el]
            c_la_c_el += N_c.T @ Ci_inv @ N_c

        return c_la_c_el

    # la_c = K_c @ l_c
    def la_c(self, t, q, u):
        la_c = np.zeros(self.nla_c)
        for el in range(self.nelement):
            qe = q[self.elDOF[el]]
            ue = u[self.elDOF_u[el]]
            la_cDOF = self.elDOF_la_c[el]
            la_c[la_cDOF] = self.la_c_el(qe, ue, el)

        return la_c

    def la_c_el(self, qe, ue, el):
        l_c = self.internal_master(qe, el)[0][self.cDOF_el]
        return np.linalg.solve(self.K_c_inv[el], l_c)

    # c = l_c - K_c^{-1} @ la_c
    def c(self, t, q, u, la_c):
        c = np.zeros(self.nla_c, dtype=np.common_type(q, u, la_c))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_la_c = self.elDOF_la_c[el]
            # TODO: is there a faster version?
            l_c = self.internal_master(q[elDOF], el)[0][self.cDOF_el]
            c[elDOF_la_c] = l_c - self.K_c_inv[el] @ la_c[elDOF_la_c]
        return c

    def c_q(self, t, q, u, la_c):
        coo = CooMatrix((self.nla_c, self.nq))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_la_c = self.elDOF_la_c[el]
            coo[elDOF_la_c, elDOF] = self.internal_master(q[elDOF], el)[1][self.cDOF_el]
        return coo

    def c_el_qe(self, t, q, u, la_c): ...

    # generalized force direction
    def W_c(self, t, q):
        coo = CooMatrix((self.nu, self.nla_c))
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            elDOF_u = self.elDOF_u[el]
            elDOF_la_c = self.elDOF_la_c[el]
            W_compliance_el = self.internal_master(q[elDOF], el)[2]
            coo[elDOF_u, elDOF_la_c] = W_compliance_el[:, self.cDOF_el]
        return coo

    def Wla_c_q(self, t, q, la_c): ...

    ###############
    # constraints #
    ###############
    # TODO: add constraints only when constrained

    ########################
    # evaluation functions #
    ########################
    def _eval_stresses(self, t, q, la_c, la_g, xi, el): ...

    def _eval_straints(self, t, q, la_c, la_g, xi, el): ...


def make_BoostedCosseratRod(
    *,
    polynomial_degree=None,
    constraints=None,
    nquadrature_int=None,
    nquadrature_dyn=None,
):
    if constraints is not None:
        if not (
            (np.array(constraints) >= 0).all() & (np.array(constraints) <= 5).all()
        ):
            raise ValueError("constraint values must between 0 and 5")

    polynomial_degree = 2 if polynomial_degree is None else polynomial_degree

    nquadrature_int = polynomial_degree if nquadrature_int == None else nquadrature_int
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
                constraints,
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
        def qu_order(
            q=None,
            u=None,
            old_to_new=True,
        ):
            if q is not None:
                nnodes = len(q) / 7
                assert int(nnodes) == nnodes
                nnodes = int(nnodes)
                if old_to_new:
                    r = q[: 3 * nnodes].reshape((3, nnodes))
                    p = q[3 * nnodes :].reshape((4, nnodes))

                    rP = np.array(
                        [np.concatenate([r[:, i], p[:, i]]) for i in range(nnodes)]
                    )
                    q = rP.reshape(-1, order="C")
                else:
                    print("not implemented yet!")

                if u is None:
                    return q

            if u is not None:
                print("not implemented yet")

            return q, u

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
            rP = np.zeros((7, nnodes), dtype=float)
            for i in range(nnodes):
                rP[:3, i] = r_OP0 + A_IB0 @ r_OP[:, i]
                rP[3:, i] = p

            return rP.reshape(-1, order="C")

        # TODO: also copy&paste the other configurations
        # The order is the same!

    return BoostedCosseratRod_PetrovGalerkin
