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

from ._base_export import RodExportBase
from ._cross_section import CrossSectionInertias
from .discretization.lagrange import LagrangeKnotVector
from .discretization.mesh1D import Mesh1D

zeros3 = np.zeros(3, dtype=float)
eye3 = np.eye(3, dtype=float)

# TODO: remove eye6
eye6 = np.eye(6, dtype=float)


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
        if cross_section_inertias == None:
            self.cross_section_inertias = CrossSectionInertias()
        else:
            #
            self.cross_section_inertias = cross_section_inertias
            self.h = self._h
            self.h_u = self._h_u

        # TODO: rename these in idx_c and idx_g
        self.idx_impressed = np.setdiff1d(np.arange(6), np.atleast_1d(constraints))
        self.idx_constrained = np.setdiff1d(
            np.arange(6), np.atleast_1d(self.idx_impressed)
        )

        # TODO: remove these as only used once
        self.n_constrained = len(self.idx_constrained)
        self.n_impressed = len(self.idx_impressed)

        self.name = "Cosserat_rod" if name is None else name

        self.nelement = nelement
        self.polynomial_degree = polynomial_degree

        # TODO: also add posibility to switch from Gauss to Lobatto/Trapezoidal
        self.nquadrature_int = nquadrature_int
        self.nquadrature_dyn = nquadrature_dyn

        # TODO: check where else this is used
        self.nquadrature = nquadrature_dyn

        self.knot_vector = LagrangeKnotVector(polynomial_degree, nelement)

        # TODO: update mesh
        # mesh for interpolation of (r, P) and (v, omega)
        mesh_kin = Mesh1D(
            knot_vector=self.knot_vector,
            nquadrature=1,  # dummy value
            dim_q=7,
            derivative_order=1,
            basis="Lagrange",
            dim_u=6,
        )
        self.mesh_kin = mesh_kin

        # total number of nodes and per element
        self.nnodes = mesh_kin.nnodes
        self.nnodes_element = mesh_kin.nnodes_per_element

        # TODO: this should be in the knot vector!
        self.xis_nodes = np.linspace(0, 1, self.nnodes)
        self.xis_element_boundaries = self.xis_nodes[::polynomial_degree]

        # total number of generalized position and velocity coordinates
        self.nq = mesh_kin.nq
        self.nu = mesh_kin.nu

        # TODO: remove
        self.nq_r = int(self.nq / 7 * 3)
        self.nu_r = int(self.nu / 6 * 3)

        # TODO: remove after complete vectorization
        # number of generalized position and velocity coordinates per element
        self.nq_element = mesh_kin.nq_per_element
        self.nu_element = mesh_kin.nu_per_element

        # TODO: remove after complete vectorization
        self.nq_element_r = int(self.nq_element / 7 * 3)
        self.nu_element_r = int(self.nu_element / 6 * 3)

        # TODO: update elDOFs for new ordering
        # global element connectivity
        # qe = q[elDOF[e]] "q^e = C_q,e q"
        self.elDOF = mesh_kin.elDOF
        # ue = u[elDOF_u[e]] "u^e = C_u,e u"
        self.elDOF_u = mesh_kin.elDOF_u

        # TODO: update nodalDOFs for new ordering and remove for r and P, v and o
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
        # (r_OP, P)_i = qe[nodalDOF_element[i]]
        self.nodalDOF_element = mesh_kin.nodalDOF_element
        # (v_P, P)_i^e = Cu_i^e * C_u,e u = Cu_r,i^e * u^e
        # (v_P, P)_i = ue[nodalDOF_element_u[i]]
        self.nodalDOF_element_u = mesh_kin.nodalDOF_element_u

        # TODO: clean up functions, seems handy to keep them for partial derivatives
        # evaluate shape functions at specific xi
        self.basis_functions = mesh_kin.eval_basis
        self.Nq = mesh_kin.eval_basis_matrix_q
        self.Nu = mesh_kin.eval_basis_matrix_u

        #####################
        # quadrature points #
        #####################
        # TODO: remove these here after new ordering works
        self.qp_int, self.qw_int = mesh_kin.quadrature_points(self.nquadrature_int)
        self.qp_dyn, self.qw_dyn = mesh_kin.quadrature_points(self.nquadrature_dyn)

        #####
        # TODO: remove these here after new ordering works
        #####
        # shape functions and their first derivatives
        # for quadrature points of internal virtual work
        N_mtx_q, N_mtx_u = mesh_kin.shape_functions_matrix(self.nquadrature_int, 1)
        self.Nq_int, self.Nq_xi_int = N_mtx_q
        self.Nu_int, self.Nu_xi_int = N_mtx_u

        self.Nv_int = self.Nu_int[:, :, :3, : self.nu_element_r]
        self.No_int = self.Nu_int[:, :, 3:, self.nu_element_r :]

        # quadrature
        self.nquadrature_int_total = self.nquadrature_int * self.nelement
        self.qp_int_vec = self.qp_int.reshape(-1)
        self.qw_int_vec = self.qw_int.reshape(-1)
        self.N_int, self.N_xi_int = mesh_kin.shape_functions_matrix_new(
            self.nquadrature_int, 1
        )

        # for quadrature points of dynamic virtual work
        # TODO: remove these after vectorization of M and h/f_gyr
        Nq_dyn, Nu_dyn = mesh_kin.shape_functions_matrix(self.nquadrature_dyn, 1)
        self.Nq_dyn, self.Nq_xi_dyn = Nq_dyn
        self.Nu_dyn = Nu_dyn[0]

        self.Nv_dyn = self.Nu_dyn[:, :, :3, : self.nu_element_r]
        self.No_dyn = self.Nu_dyn[:, :, 3:, self.nu_element_r :]

        # quadrature
        # TODO: allow for trapezoidal rule
        self.nquadrature_dyn_total = self.nquadrature_dyn * self.nelement
        self.qp_dyn_vec = self.qp_dyn.reshape(-1)
        self.qw_dyn_vec = self.qw_dyn.reshape(-1)
        self.N_dyn, self.N_xi_dyn = mesh_kin.shape_functions_matrix_new(
            self.nquadrature_dyn, 1
        )

        # referential generalized position coordinates, initial generalized
        # position and velocity coordinates
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else u0

        ##############
        # new ordering
        ##############
        Q = Q.reshape(-1, self.nnodes).reshape(-1, order="F")
        self.q0 = self.q0.reshape(-1, self.nnodes).reshape(-1, order="F")
        self.u0 = self.u0.reshape(-1, self.nnodes).reshape(-1, order="F")

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
        # TODO: check all from below here if used
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

        # TODO: check what actually is used from here and prepare for hard constraint

        # total number of nodes
        self.nnodes_sigma = mesh_rcs.nnodes

        # number of nodes per element
        self.nnodes_element_sigma = mesh_rcs.nnodes_per_element

        # total number of compliance coordinates
        self.nla_sigma = mesh_rcs.nq
        if (nla_c := int(mesh_rcs.nq / 6 * self.n_impressed)) > 0:
            self.nla_c = nla_c
        if (nla_g := int(mesh_rcs.nq / 6 * self.n_constrained)) > 0:
            self.nla_g = nla_g

        # number of compliance coordinates per element
        self.nla_sigma_element = mesh_rcs.nq_per_element

        # global element connectivity for compliance coordinates
        self.elDOF_la_c = mesh_rcs.elDOF

        # global nodal connectivity
        self.nodalDOF_la_c = mesh_rcs.nodalDOF

        # nodal connectivity on element level
        self.nodalDOF_element_la_c = mesh_rcs.nodalDOF_element

        # evaluate shape functions at specific xi
        self.basis_functions_la_c = mesh_rcs.eval_basis

        self.Nc = mesh_rcs.shape_functions_matrix(self.nquadrature_int, 0)[0][0]
        self.Nc_int = mesh_rcs.shape_functions_matrix_new(self.nquadrature_int, 0)[0]

        # TODO: this splits c and g from l_c_el
        self.cDOF_el = np.arange(self.nla_sigma_element)

        ################
        # assemble matrices for block structure
        ################
        self.NN_int_bd = self.create_block_dict(self.N_int, self.N_int)
        self.NN_xi_int_bd = self.create_block_dict(self.N_int, self.N_xi_int)
        self.N_xiN_int_bd = self.create_block_dict(self.N_xi_int, self.N_int)
        self.N_xiN_xi_int_bd = self.create_block_dict(self.N_xi_int, self.N_xi_int)

        self.NcNc_int_bd = self.create_block_dict(self.Nc_int, self.Nc_int)

        self.NcN_int_bd = self.create_block_dict(self.Nc_int, self.N_int)
        self.NNc_int_bd = self.create_block_dict(self.N_int, self.Nc_int)
        self.NcN_xi_int_bd = self.create_block_dict(self.Nc_int, self.N_xi_int)
        self.N_xiNc_int_bd = self.create_block_dict(self.N_xi_int, self.Nc_int)

        self.NN_dyn_bd = self.create_block_dict(self.N_dyn, self.N_dyn)

        ################
        # permutations #
        ################
        # from ordering componentwise (old) to nodewise (new)
        # gen. positions q
        iq = np.arange(self.nnodes)[:, None]
        jq = np.arange(7)[None, :]

        idxq_new = jq + iq * 7
        idxq_old = iq + jq * self.nnodes

        self.permutation_comp2node_q = idxq_old.ravel()[np.argsort(idxq_new.ravel())]
        self.permutation_node2comp_q = idxq_new.ravel()[np.argsort(idxq_old.ravel())]

        iq = np.arange(self.nnodes_element)[:, None]
        jq = np.arange(7)[None, :]

        idxq_new = jq + iq * 7
        idxq_old = iq + jq * self.nnodes_element

        self.permutation_comp2node_q_el = idxq_old.ravel()[np.argsort(idxq_new.ravel())]
        self.permutation_node2comp_q_el = idxq_new.ravel()[np.argsort(idxq_old.ravel())]

        # gen. velocities u
        iu = np.arange(self.nnodes)[:, None]
        ju = np.arange(6)[None, :]

        idxu_new = ju + iu * 6
        idxu_old = iu + ju * self.nnodes

        self.permutation_comp2node_u = idxu_old.ravel()[np.argsort(idxu_new.ravel())]
        self.permutation_node2comp_u = idxu_new.ravel()[np.argsort(idxu_old.ravel())]

        iu = np.arange(self.nnodes_element)[:, None]
        ju = np.arange(6)[None, :]

        idxu_new = ju + iu * 6
        idxu_old = iu + ju * self.nnodes_element

        self.permutation_comp2node_u_el = idxu_old.ravel()[np.argsort(idxu_new.ravel())]
        self.permutation_node2comp_u_el = idxu_new.ravel()[np.argsort(idxu_old.ravel())]

        # generalized forces la_c
        ic = np.arange(self.nnodes_sigma)[:, None]
        jc = np.arange(6)[None, :]

        idxc_new = jc + ic * 6
        idxc_old = ic + jc * self.nnodes_sigma

        self.permutation_comp2node_c = idxc_old.ravel()[np.argsort(idxc_new.ravel())]
        self.permutation_node2comp_c = idxc_new.ravel()[np.argsort(idxc_old.ravel())]

        ic = np.arange(self.nnodes_element_sigma)[:, None]
        jc = np.arange(6)[None, :]

        idxc_new = jc + ic * 6
        idxc_old = ic + jc * self.nnodes_element_sigma

        self.permutation_comp2node_c_el = idxc_old.ravel()[np.argsort(idxc_new.ravel())]
        self.permutation_node2comp_c_el = idxc_new.ravel()[np.argsort(idxc_old.ravel())]

        # F: qnodes = q_cardillo.reshape(nnodes, -1)
        self.__current_order = "C"
        # "C" is default

        # TODO: do inner dict with enum as key or use a class instead of dict
        self.interaction_points: dict[float, dict[str, np.ndarray]] = {}

        self.set_reference_strains(Q)

        ##########
        # caches #
        ##########
        # TODO: check again which caches are usefull
        ninteractions = 2
        # TODO: get this number based on the number of interactions
        self._cache_f_gyr = LRUCache(self.nquadrature_dyn * nelement)
        self._cache_internal = LRUCache(self.nquadrature_int * nelement)

        self._cache_element_number = LRUCache(50 * ninteractions)
        self._cache_eval_r_A = LRUCache(ninteractions)
        self._cache_velocity_rotational = LRUCache(ninteractions)

        # pre-evaluated zeros
        self.zero_3_nqe = np.zeros((3, self.nq_element), dtype=float)
        self.zero_3_nue = np.zeros((3, self.nu_element), dtype=float)

        #######
        # use new, old or both
        #######
        # compliance
        # self.c = self.c_compare
        # self.W_c = self.W_c_compare

        self.c = self.c
        self.W_c = self.W_c

        print("init done")

    @property
    def current_order(self):
        # print("current order used")
        return self.__current_order

    def create_block_dict(self, Na, Nb):
        block_dict = {}  # key = (row_block, col_block), value = list of (i, value)
        assert Na.shape[0] == Nb.shape[0]
        neval = Na.shape[0]
        for i in range(neval):
            Ni_outer = (Na[i][:, None] @ Nb[i][None, :]).tocoo()
            for r, c, N in zip(Ni_outer.row, Ni_outer.col, Ni_outer.data):
                block_dict.setdefault((r, c), []).append((i, N))

        block_positions = list(block_dict.keys())
        nblocks = len(block_positions)
        block_rows, block_cols = np.array(block_positions).T

        weights_matrix = np.zeros((nblocks, neval))
        for b, pos in enumerate(block_positions):
            for i, N in block_dict[pos]:
                weights_matrix[b, i] = N

        # TODO: check if we can avoid ordering here by ordering above the N/N_xi
        # TODO: do we even have to order?
        order = np.lexsort((block_cols, block_rows))
        # assert (order == np.arange(len(order))).all()
        if not (order == np.arange(len(order))).all():
            print("Reordered!")
            block_rows = block_rows[order]
            block_cols = block_cols[order]
            # blocks = blocks[order]

        # TODO: can we find an explicit expression for indptr?
        # TODO: check if it is Na.shape[1] + 1 or Nb.shape[1] + 1
        indptr = np.zeros(Na.shape[1] + 1, dtype=int)
        np.add.at(indptr, block_rows + 1, 1)
        indptr = np.cumsum(indptr)

        return dict(
            weights_matrix=weights_matrix,
            block_cols=block_cols,
            indptr=indptr,
            order=order,
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

        if True:
            # TODO: remove all from below f_gyr must be updated!

            # precompute values of the reference configuration in order to save
            # computation time J in Harsch2020b (5)
            self.J_int = np.zeros((self.nelement, self.nquadrature_int), dtype=float)
            self.J_dyn = np.zeros((self.nelement, self.nquadrature_dyn), dtype=float)
            # strains of the reference configuration
            self.epsilon0 = np.zeros(
                (self.nelement, self.nquadrature_int, 6), dtype=float
            )
            self.epsilon0_bar = np.zeros(
                (self.nelement, self.nquadrature_int, 6), dtype=float
            )

            for el in range(self.nelement):
                qe = self.Q[self.permutation_node2comp_q][self.elDOF[el]]

                for i in range(self.nquadrature_int):
                    # evaluate required quantities
                    qpi = self.qp_int[el, i]
                    N = self.Nq_int[el, i]
                    N_xi = self.Nq_xi_int[el, i]
                    epsilon_bar = self._eval_internal(qpi, N, N_xi, qe)[0][2]

                    # save length of reference tangential vector and strain
                    self.J_int[el, i] = norm(epsilon_bar[:3])
                    self.epsilon0[el, i] = epsilon_bar / self.J_int[el, i]
                    self.epsilon0_bar[el, i] = epsilon_bar

                for i in range(self.nquadrature_dyn):
                    # evaluate required quantities
                    qpi = self.qp_dyn[el, i]
                    N = self.Nq_dyn[el, i]
                    N_xi = self.Nq_xi_dyn[el, i]
                    self.J_dyn[el, i] = self.compute_J(qpi, N, N_xi, qe)

    # TODO: maybe it is more practical to set up a function to return
    # TODO: use it with line distributed force
    # qp, qw, J, [(Nq, Nq_xi) or (Nu, Nu_xi)] for given number of quadrature points
    def get_quadrature(self, nquadrature, deriv, field):
        """Number of quadrature point, how many derivatives, and what filed (r, P, q, v, Om, u, n, m, la_c)"""

    def compute_J(self, xi, N, N_xi, qe):
        epsilon_bar = self._eval_internal(xi, N, N_xi, qe)[0][2]

        # length of reference tangential vector
        return norm(epsilon_bar[:3])

    # TODO: When are these functions called? Do we need and can we speed them up?
    def element_interval(self, el):
        return self.knot_vector.element_interval(el)

    ############################
    # export of centerline nodes
    ############################
    def nodes(self, qsystem):
        """Returns nodal position coordinates"""
        qbody = qsystem[self.qDOF]
        qnodesT = qbody.reshape(self.nnodes, -1, order="F")
        return qnodesT[:3]

    def nodalFrames(self, q, elementwise=False):
        """Returns nodal positions and nodal directors.
        If elementwise==True : returned arrays are each of shape [nnodes, 3]
        If elementwise==False : returned arrays are each of shape [nelements, nnodes_per_element, 3]
        """
        # TODO: use vectorization

        q_body = q[self.qDOF][self.permutation_node2comp_q]
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

    # def frames(self, qsystem, num=10):
    #     # TODO: update this with new mesh
    #     qbody = qsystem[self.qDOF]
    #     qnodes = qbody.reshape(self.nnodes, -1)

    #     # maybe cach the N matrix
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

    ##################
    # eval functions #
    ##################
    # @cachedmethod(
    #     lambda self: self._cache_eval_r_A,
    #     key=lambda self, xi, Nq, qe: hashkey(xi, *qe),
    # )
    def _eval_r_A(self, xi, Nq, qe):
        rP = Nq @ qe
        A_IB = Exp_SO3_quat(rP[3:])
        A_IB_qe = Exp_SO3_quat_P(rP[3:], normalize=True) @ Nq[3:]
        return (rP[:3], A_IB), (Nq[:3], A_IB_qe)

    def _veval_v_Om(self, xi, Nu, ue):
        vOm = Nu @ ue
        return vOm[:3], vOm[3:]

    def _eval_internal(self, xi, Nq, Nq_xi, qe):
        # eval
        rP = Nq @ qe
        rP_xi = Nq_xi @ qe
        A_IB = Exp_SO3_quat(rP[3:])
        T = T_SO3_quat(rP[3:], normalize=True)
        B_gamma = A_IB.T @ rP_xi[:3]
        B_kappa = T @ rP_xi[3:]
        eps = np.array([*B_gamma, *B_kappa])

        # deval
        A_IB_qe = Exp_SO3_quat_P(rP[3:], normalize=True) @ Nq[3:]
        # TODO: think of implementing (T_SO3_quat(P) @ Q)_P for kappa
        # (ca. 10% speed up per call)
        # TP_xi_P = rP_xi[3:] @ T_SO3_quat_P(rP[3:], normalize=True)
        TP_xi_P = T_SO3_quat_Q_P(rP[3:], rP_xi[3:], normalize=True)
        B_gamma_bar_qe = ax2skew(B_gamma) @ T @ Nq[3:] + A_IB.T @ Nq_xi[:3]
        B_kappa_bar_qe = TP_xi_P @ Nq[3:] + T @ Nq_xi[3:]
        eps_qe = np.vstack([B_gamma_bar_qe, B_kappa_bar_qe])
        return (rP[:3], A_IB, eps), (Nq[:3], A_IB_qe, eps_qe)

    #########################################
    # equations of motion
    #########################################
    def M(self, t, q):
        return self.__M

    def _M_coo(self):
        weights_matrix = self.NN_dyn_bd["weights_matrix"]
        block_cols = self.NN_dyn_bd["block_cols"]
        indptr = self.NN_dyn_bd["indptr"]
        order = self.NN_dyn_bd["order"]

        # TODO: make this sparse?
        M_qp = np.empty((self.nquadrature_dyn_total, 6, 6))
        M_qp[:, :3, :3] = eye3 * self.cross_section_inertias.A_rho0
        M_qp[:, :3, 3:] = 0.0
        M_qp[:, 3:, :3] = 0.0
        M_qp[:, 3:, 3:] = self.cross_section_inertias.B_I_rho0

        blocks = np.einsum(
            "bi,i,ikl->bkl",
            weights_matrix,  # [iBlock, qpi]
            self.qw_dyn_vec * self.J_dyn_vec,  # [qpi]
            M_qp,  # [qpi, uDOF, la_cDOF]
        )

        self.__M = bsr_array(
            (blocks[order], block_cols, indptr),
            shape=(self.nu, self.nu),
            blocksize=(6, 6),
        )

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

    # TODO: vectorize
    def _h_u(self, t, q, u):
        coo = CooMatrix((self.nu, self.nu))
        warn("No gyroscopic forces .. h_u!")
        return coo
        for el in range(self.nelement):
            elDOF_u = self.elDOF_u[el][self.nu_element_r :]
            coo[elDOF_u, elDOF_u] = -self.f_gyr_master(u[elDOF_u], el)[1]
        return coo

    # @cachedmethod(
    #     lambda self: self._cache_f_gyr,
    #     key=lambda self, ue_red, el: hashkey(*ue_red, el),
    # )
    def f_gyr_master(self, ue_red, el):
        warn("Do this w/o loops")
        # only compute part with omega
        dim_red = self.nu_element - self.nu_element_r
        f_gyr_el = np.zeros(dim_red, dtype=ue_red.dtype)
        f_gyr_el_ue = np.zeros((dim_red, dim_red), dtype=ue_red.dtype)

        for i in range(self.nquadrature_dyn):
            # interpoalte angular velocity
            No = self.No_dyn[el, i]
            B_Omega = No @ ue_red
            weight = self.J_dyn[el, i] * self.qw_dyn[el, i]

            # vector of gyroscopic forces
            B_I_rho0 = self.cross_section_inertias.B_I_rho0
            f_gyr_el_p = cross3(B_Omega, B_I_rho0 @ B_Omega) * weight
            f_gyr_u_el_p = (
                (ax2skew(B_Omega) @ B_I_rho0 - ax2skew(B_I_rho0 @ B_Omega))
            ) * weight

            # multiply vector of gyroscopic forces with nodal virtual rotations
            f_gyr_el += No.T @ f_gyr_el_p
            f_gyr_el_ue += No.T @ f_gyr_u_el_p @ No

        return f_gyr_el, f_gyr_el_ue

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
        # TODO: update elDOF
        el = self.element_number(xi)
        return self.elDOF[el]

    def elDOF_P_u(self, xi):
        # TODO: update elDOF
        el = self.element_number(xi)
        return self.elDOF_u[el]

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

                # TODO: avoid using Nq and Nu from the mesh
                Nq = self.Nq(xi, el, 0)
                Nu = self.Nu(xi, el, 0)
                N = Nq[0, :3]

                # TODO: avoid this elDOF and permutation afterwards!
                qDOF = self.elDOF_P(xi)
                uDOF = self.elDOF_P_u(xi)

                Nq = Nq[:, self.permutation_comp2node_q_el]
                Nu = Nu[:, self.permutation_comp2node_u_el]

                start = (nnodes - 1) * el
                end = (nnodes - 1) * (el + 1) + 1
                qDOF = np.arange(7 * start, 7 * end)
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
                J_C_q=np.zeros((3, 6 * nnodes, 7 * nnodes)),
                B_J_R_q=np.zeros((3, 6 * nnodes, 7 * nnodes)),
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

    # TODO: remove function?
    def node_number_element(self, xi):
        """For given xi in I = [0.0, 1.0], returns element node number if xi is a node, otherwise False"""
        idx = np.where(self.xis_nodes == xi)[0]
        if len(idx) == 1:
            idx = idx[0]
            if idx == self.nnodes - 1:
                return self.polynomial_degree
            else:
                return idx % self.polynomial_degree
        else:
            return False

    # @cachedmethod(
    #     lambda self: self._cache_element_number,
    #     key=lambda self, xi: hashkey(xi),
    # )
    def element_number(self, xi):
        return np.where(self.xis_element_boundaries[:-1] <= xi)[0][-1]

    def _eval_nodal(self, qe, node):
        """returns (r_OC, A_IB), (r_OC_qe, A_IB_qe), (J_C, B_J_R)"""
        # split up nodalDOF
        nodalDOF = self.nodalDOF_element[node]
        nodalDOF_u = self.nodalDOF_element_u[node]
        nodalDOF_r = nodalDOF[:3]
        nodalDOF_P = nodalDOF[3:]
        # transformation matrix
        P = qe[nodalDOF_P]
        A_IB = Exp_SO3_quat(P, normalize=True)
        A_IB_qe = np.zeros((3, 3, self.nq_element), dtype=qe.dtype)
        A_IB_qe[:, :, nodalDOF_P] = Exp_SO3_quat_P(P, normalize=True)
        # centerline
        r_OC_qe = np.zeros((3, self.nq_element), dtype=qe.dtype)
        r_OC_qe[:, nodalDOF_r] = eye3

        # Jacobians
        J_total = np.zeros((6, self.nu_element), dtype=float)
        J_total[:, nodalDOF_u] = eye6
        return (qe[nodalDOF_r], A_IB), (r_OC_qe, A_IB_qe), (J_total[:3], J_total[3:])

    def _veval_nodal(self, ue, node):
        """if argument is ue:\n
        returns (v_C, B_Omega_IB), (J_C, B_J_R) \n
        if argument is ue_dot:\n
        returns (a_C, B_Psi_IB), (J_C, B_J_R)"""
        nodalDOF_u = self.nodalDOF_element_u[node]
        v = ue[nodalDOF_u]
        return v[:3], v[3:]

    def velocity_translational(self, qe, ue, xi, B_r_CP=zeros3):
        """returns (v_P, v_P_qe)"""

        if not xi in self.interaction_points.keys():
            warn("xi was not initialized")
            _ = self.local_qDOF_P(xi)
        else:
            point_dict = self.interaction_points.get(xi)
            Nq = point_dict["Nq"]
            Nu = point_dict["Nu"]

        if (node := self.node_number_element(xi)) is not False:
            _eval, _deval, Jacobians = self._eval_nodal(qe, node)
            _veval = self._veval_nodal(ue, node)
        else:
            # el = self.element_number(xi)
            # Nq = self.Nq(xi, el, 0)
            _eval, _deval = self._eval_r_A(xi, Nq, qe)
            # Nu = self.Nu(xi, el, 0)
            _veval = self._veval_v_Om(xi, Nu, ue)

        if B_r_CP @ B_r_CP > 0.0:
            B_v_CP = cross3(_veval[1], B_r_CP)
            v_P = _veval[0] + _eval[1] @ B_v_CP
            # TODO: accelerate this
            v_P_qe = np.einsum("ijk,j->ik", _deval[1], B_v_CP)
            return v_P, v_P_qe
        else:
            return _veval[0], self.zero_3_nqe

    # TODO: cache this with *qe, *ue, *ue_dot, *B_r_CP, xi
    # TODO: cache for integrators with acceleration level
    def acceleration_translational(self, qe, ue, ue_dot, xi, B_r_CP=zeros3):
        """returns (a_P, a_P_qe, a_P_ue)"""
        print(f"acceleration_translational")
        if (node := self.node_number_element(xi)) is not False:
            _eval, _deval, Jacobians = self._eval_nodal(qe, node)
            _veval = self._veval_nodal(ue, node)
            _aeval = self._veval_nodal(ue_dot, node)
            B_J_R = Jacobians[1]
        else:
            el = self.element_number(xi)
            Nq = self.Nq(xi, el, 0)
            _eval, _deval = self._eval_r_A(xi, Nq, qe)
            Nu = self.Nu(xi, el, 0)
            _veval = self._veval_v_Om(xi, Nu, ue)
            _aeval = self._veval_v_Om(xi, Nu, ue_dot)
            B_J_R = Nu[3:]

        if B_r_CP @ B_r_CP > 0.0:
            A_IB = _eval[1]
            B_Omega_IB = _veval[1]
            B_a_CP = cross3(B_Omega_IB, cross3(B_Omega_IB, B_r_CP)) + cross3(
                _aeval[1], B_r_CP
            )
            a_P = _aeval[0] + A_IB @ B_a_CP
            # TODO: accelerate this
            a_P_qe = np.einsum("ijk,j->ik", _deval[1], B_a_CP)
            a_P_ue = (
                -A_IB
                @ (
                    ax2skew(cross3(B_Omega_IB, B_r_CP))
                    + ax2skew(B_Omega_IB) @ ax2skew(B_r_CP)
                )
            ) @ B_J_R
            return a_P, a_P_qe, a_P_ue
        else:
            return _aeval[0], self.zero_3_nqe, self.zero_3_nue

    # @cachedmethod(
    #     lambda self: self._cache_velocity_rotational,
    #     key=lambda self, ue, xi: hashkey(*ue, xi),
    # )
    def velocity_rotational(self, ue, xi):
        """returns (B_Omega_IB, B_Omega_IB_qe)"""
        if (node := self.node_number_element(xi)) is not False:
            _veval = self._veval_nodal(ue, node)
        else:
            el = self.element_number(xi)
            Nu = self.Nu(xi, el, 0)
            _veval = self._veval_v_Om(xi, Nu, ue)

        return _veval[1]

    # cardillo functions
    def r_OP(self, t, qe, xi, B_r_CP=zeros3):
        point_dict = self.get_interaction_point(xi)
        N = point_dict["N"]
        qnodes = qe.reshape(point_dict["nnodes"], -1, order=self.current_order)
        if B_r_CP @ B_r_CP == 0.0:
            return N @ qnodes[:, :3]
        rP = N @ qnodes
        return rP[:3] + self._A_IB(rP[3:]) @ B_r_CP

    def r_OP_q(self, t, qe, xi, B_r_CP=zeros3):
        point_dict = self.get_interaction_point(xi)
        if B_r_CP @ B_r_CP == 0.0:
            return point_dict["r_q"]
        return point_dict["r_q"] + np.einsum(
            "ijk,j->ik", self.A_IB_q(t, qe, xi), B_r_CP
        )

    def J_P(self, t, qe, xi, B_r_CP=zeros3):
        point_dict = self.get_interaction_point(xi)
        if B_r_CP @ B_r_CP == 0.0:
            return point_dict["J_C"]

        qnodes = qe.reshape(point_dict["nnodes"], -1, order=self.current_order)
        P = point_dict["N"] @ qnodes[:, 3:]
        return point_dict["J_C"] - self._A_IB(P) @ ax2skew(B_r_CP) @ point_dict["B_J_R"]

    def J_P_q(self, t, qe, xi, B_r_CP=zeros3):
        point_dict = self.get_interaction_point(xi)
        if B_r_CP @ B_r_CP == 0.0:
            # this is zero
            return point_dict["J_C_q"]

        B_J_CP = ax2skew(-B_r_CP) @ point_dict["B_J_R"]
        return np.einsum("ijk, jl -> ilk", self.A_IB_q(t, qe, xi), B_J_CP)

    def v_P(self, t, qe, ue, xi, B_r_CP=zeros3):
        point_dict = self.get_interaction_point(xi)
        N = point_dict["N"]
        unodes = ue.reshape(point_dict["nnodes"], -1, order=self.current_order)
        if B_r_CP @ B_r_CP == 0.0:
            return N @ unodes[:, :3]
        qnodes = qe.reshape(point_dict["nnodes"], -1, order=self.current_order)
        P = N @ qnodes[:, 3:]
        vO = N @ unodes
        return vO[:3] + self._A_IB(P) @ (cross3(vO[3:], B_r_CP))

    def v_P_q(self, t, qe, ue, xi, B_r_CP=zeros3):
        # TODO
        print("v_P_q")

        # # TODO: replace all cross3 by np.cross
        # B_v_CP = cross3(_veval[1], B_r_CP)
        # v_P = _veval[0] + _eval[1] @ B_v_CP
        # # TODO: accelerate this
        # v_P_qe = np.einsum("ijk,j->ik", _deval[1], B_v_CP)
        return self.velocity_translational(qe, ue, xi, B_r_CP)[1]

    def a_P(self, t, qe, ue, ue_dot, xi, B_r_CP=zeros3):
        print("a_P")
        return self.acceleration_translational(qe, ue, ue_dot, xi, B_r_CP)[0]

    def a_P_q(self, t, qe, ue, ue_dot, xi, B_r_CP=zeros3):
        print("a_P_q")
        return self.acceleration_translational(qe, ue, ue_dot, xi, B_r_CP)[1]

    def a_P_u(self, t, qe, ue, ue_dot, xi, B_r_CP=zeros3):
        print("a_P_u")
        return self.acceleration_translational(qe, ue, ue_dot, xi, B_r_CP)[2]

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

    def A_IB(self, t, qe, xi):
        point_dict = self.get_interaction_point(xi)
        N = point_dict["N"]
        qnodes = qe.reshape(point_dict["nnodes"], -1, order=self.current_order)
        return self._A_IB(N @ qnodes[:, 3:])

    def A_IB_q(self, t, qe, xi):
        point_dict = self.get_interaction_point(xi)
        N = point_dict["N"]
        qnodes = qe.reshape(point_dict["nnodes"], -1, order=self.current_order)
        P = N @ qnodes[:, 3:]
        return self._A_IB_P(P) @ point_dict["P_q"]

    def B_J_R(self, t, qe, xi):
        point_dict = self.get_interaction_point(xi)
        return point_dict["B_J_R"]

    def B_J_R_q(self, t, qe, xi):
        point_dict = self.get_interaction_point(xi)
        return point_dict["B_J_R_q"]

    def B_Omega(self, t, qe, ue, xi):
        point_dict = self.get_interaction_point(xi)
        N = point_dict["N"]
        unodes = ue.reshape(point_dict["nnodes"], -1, order=self.current_order)
        return N @ unodes[:, 3:]

    def B_Omega_q(self, t, qe, ue, xi):
        print("B_Omega_q")
        # TODO: size must be (3, 7*point_dict["nnodes"])
        return self.zero_3_nqe

    def B_Psi(self, t, qe, ue, ue_dot, xi):
        print("B_Psi")
        return self.velocity_rotational(ue_dot, xi)

    def B_Psi_q(self, t, qe, ue, ue_dot, xi):
        print("B_Psi_q")
        # TODO: size must be (3, 7*point_dict["nnodes"])
        return self.zero_3_nqe

    def B_Psi_u(self, t, qe, ue, ue_dot, xi):
        print("B_Psi_u")
        # TODO: size must be (3, 6*point_dict["nnodes"])
        return self.zero_3_nue

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


    ##############
    # compliance #
    ##############
    # TODO: check +- or remove completely
    # c = l_c - K_c^{-1} @ la_c

    # la_c = -c_la_c @ c(q, u, 0)

    # K_c^{-1}
    def c_la_c(self):
        return self.__cla_c

    def la_c(self, t, q, u):
        return self.__cla_c_inv @ self.c_sigma(q, u)

    def c_q(self, t, q, u, la_c):
        # TODO: return sparse
        return -self.c_sigma_q(q, u).toarray()

    ########################
    # vectorized functions #
    ########################
    def c_sigma(self, q, u):
        _, B_gamma_bar, B_kappa_bar = self._eval_internal_vec(
            self.N_int, self.N_xi_int, q
        )

        epsilon_bar = np.empty((self.nquadrature_int_total, 6))
        epsilon_bar[:, :3] = B_gamma_bar - self.B_gamma0_bar
        epsilon_bar[:, 3:] = B_kappa_bar - self.B_kappa0_bar

        l = self.Nc_int.T @ (epsilon_bar * self.qw_int_vec[:, None])

        # l_c = l[:, self.eps_c].reshape(-1)
        # l_g = l[:, self.eps_g].reshape(-1)
        return l.reshape(-1)

    def c(self, t, q, u, la_c):
        return self.__cla_c @ la_c - self.c_sigma(q, u)

    def c_sigma_q(self, q, u):
        # TODO: combine these!
        weights_matrix_N = self.NcN_int_bd["weights_matrix"]
        block_cols_N = self.NcN_int_bd["block_cols"]
        indptr_N = self.NcN_int_bd["indptr"]
        order_N = self.NcN_int_bd["order"]

        weights_matrix_N_xi = self.NcN_xi_int_bd["weights_matrix"]
        block_cols_N_xi = self.NcN_xi_int_bd["block_cols"]
        indptr_N_xi = self.NcN_xi_int_bd["indptr"]
        order_N_xi = self.NcN_xi_int_bd["order"]

        # compute W_sigma
        A_IB, T, B_gamma_bar_P, B_kappa_bar_P = self._eval_internal_vec(
            self.N_int, self.N_xi_int, q, deval=True
        )

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

        blocks_N = np.einsum(
            "bi,i,ikl->bkl",
            weights_matrix_N,  # [iBlock, qpi]
            self.qw_int_vec,  # [qpi]
            c_sigma_q_qp[0],  # [qpi, la_cDOF, qDOF]
        )
        blocks_N_xi = np.einsum(
            "bi,i,ikl->bkl",
            weights_matrix_N_xi,  # [iBlock, qpi]
            self.qw_int_vec,  # [qpi]
            c_sigma_q_qp[1],  # [qpi, la_cDOF, qDOF]
        )

        c_sigma_q_coo = bsr_array(
            (blocks_N[order_N], block_cols_N, indptr_N),
            shape=(self.nla_c, self.nq),
            blocksize=(6, 7),
        ) + bsr_array(
            (blocks_N_xi[order_N_xi], block_cols_N_xi, indptr_N_xi),
            shape=(self.nla_c, self.nq),
            blocksize=(6, 7),
        )
        return c_sigma_q_coo

    def _c_la_c_coo(self):
        weights_matrix = self.NcNc_int_bd["weights_matrix"]
        block_cols = self.NcNc_int_bd["block_cols"]
        indptr = self.NcNc_int_bd["indptr"]
        order = self.NcNc_int_bd["order"]

        # TODO: assuming prismatic rods, we would have to include the different stiffnesses here
        blocks = np.einsum(
            "bi,i,ikl->bkl",
            weights_matrix,  # [iBlock, qpi]
            self.qw_int_vec * self.J_int_vec,  # [qpi]
            self.material_model.C_inv[None, :, :],  # [qpi, uDOF, la_cDOF]
        )

        c_la_c = bsr_array(
            (blocks[order], block_cols, indptr),
            shape=(self.nla_c, self.nla_c),
            blocksize=(6, 6),
        )
        self.__cla_c = c_la_c
        self.__cla_c_inv = spsolve(
            self.c_la_c().tocsc(), eye_array(self.nla_c, format="csc")
        )
        return c_la_c

    def W_sigma(self, q):
        # TODO: combine these!
        weights_matrix_N = self.NNc_int_bd["weights_matrix"]
        block_cols_N = self.NNc_int_bd["block_cols"]
        indptr_N = self.NNc_int_bd["indptr"]
        order_N = self.NNc_int_bd["order"]

        weights_matrix_N_xi = self.N_xiNc_int_bd["weights_matrix"]
        block_cols_N_xi = self.N_xiNc_int_bd["block_cols"]
        indptr_N_xi = self.N_xiNc_int_bd["indptr"]
        order_N_xi = self.N_xiNc_int_bd["order"]

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

        blocks_N = np.einsum(
            "bi,i,ikl->bkl",
            weights_matrix_N,  # [iBlock, qpi]
            self.qw_int_vec,  # [qpi]
            W_sigma_qp[0],  # [qpi, uDOF, la_cDOF]
        )
        blocks_N_xi = np.einsum(
            "bi,i,ikl->bkl",
            weights_matrix_N_xi,  # [iBlock, qpi]
            self.qw_int_vec,  # [qpi]
            W_sigma_qp[1],  # [qpi, uDOF, la_cDOF]
        )

        W_coo = bsr_array(
            (blocks_N[order_N], block_cols_N, indptr_N),
            shape=(self.nu, self.nla_c),
            blocksize=(6, 6),
        ) + bsr_array(
            (blocks_N_xi[order_N_xi], block_cols_N_xi, indptr_N_xi),
            shape=(self.nu, self.nla_c),
            blocksize=(6, 6),
        )
        return W_coo

        # TODO: figure out how block_cols and indptr change
        W_c = bsr_array(
            (blocks[:, :, self.eps_c], block_cols_c, indptr_c),
            shape=(self.nu, self.nla_c),
            blocksize=(6, self.neps_c),
        )
        W_g = bsr_array(
            (blocks[:, :, self.eps_g], block_cols_g, indptr_g),
            shape=(self.nu, self.nla_g),
            blocksize=(6, self.neps_g),
        )
        return W_c, W_g

    def W_c(self, t, q):
        # TODO: return sparse
        return self.W_sigma(q).toarray()

    def Wla_sigma(self, t, q, la_sigma):
        la_sigma_nodes = la_sigma.reshape(self.nnodes_sigma, 6)
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

    def Wla_sigma_q(self, t, q, la_sigma):
        la_sigma_nodes = la_sigma.reshape(self.nnodes_sigma, 6)
        sigma_qp = self.Nc_int @ la_sigma_nodes

        # TODO: combine these!
        if True:
            weights_matrix_NN = self.NN_int_bd["weights_matrix"]
            block_cols_NN = self.NN_int_bd["block_cols"]
            indptr_NN = self.NN_int_bd["indptr"]
            order_NN = self.NN_int_bd["order"]

            weights_matrix_N_xiN = self.N_xiN_int_bd["weights_matrix"]
            block_cols_N_xiN = self.N_xiN_int_bd["block_cols"]
            indptr_N_xiN = self.N_xiN_int_bd["indptr"]
            order_N_xiN = self.N_xiN_int_bd["order"]

            weights_matrix_NN_xi = self.NN_xi_int_bd["weights_matrix"]
            block_cols_NN_xi = self.NN_xi_int_bd["block_cols"]
            indptr_NN_xi = self.NN_xi_int_bd["indptr"]
            order_NN_xi = self.NN_xi_int_bd["order"]

        # compute W_sigma
        A_IB, T, B_gamma_bar_P, B_kappa_bar_P = self._eval_internal_vec(
            self.N_int, self.N_xi_int, q, deval=True
        )

        I_n_P = np.einsum(
            "ijk,ikl->ijl",
            A_IB,
            np.cross(sigma_qp[:, :3, None], T, axisa=1, axisb=1, axisc=1),
        )

        # TODO: dense?
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

        blocks_NN = np.einsum(
            "bi,i,ikl->bkl",
            weights_matrix_NN,  # [iBlock, qpi]
            self.qw_int_vec,  # [qpi]
            Wla_sigma_qp_qbar[0],  # [qpi, uDOF, qDOF]
        )
        blocks_NN_xi = np.einsum(
            "bi,i,ikl->bkl",
            weights_matrix_NN_xi,  # [iBlock, qpi]
            self.qw_int_vec,  # [qpi]
            Wla_sigma_qp_qbar[1],  # [qpi, uDOF, qDOF]
        )
        blocks_N_xiN = np.einsum(
            "bi,i,ikl->bkl",
            weights_matrix_N_xiN,  # [iBlock, qpi]
            self.qw_int_vec,  # [qpi]
            Wla_sigma_qp_qbar[2],  # [qpi, uDOF, qDOF]
        )

        Wla_sigma_q = (
            bsr_array(
                (blocks_NN[order_NN], block_cols_NN, indptr_NN),
                shape=(self.nu, self.nq),
                blocksize=(6, 7),
            )
            + bsr_array(
                (blocks_N_xiN[order_N_xiN], block_cols_N_xiN, indptr_N_xiN),
                shape=(self.nu, self.nq),
                blocksize=(6, 7),
            )
            + bsr_array(
                (blocks_NN_xi[order_NN_xi], block_cols_NN_xi, indptr_NN_xi),
                shape=(self.nu, self.nq),
                blocksize=(6, 7),
            )
        )
        return Wla_sigma_q

    def Wla_c_q(self, t, q, la_c):
        # TODO: return sparse
        return self.Wla_sigma_q(t, q, la_c).toarray()

    ########################
    # evaluation functions #
    ########################
    def _eval_stresses(self, t, q, la_c, la_g, num_per_element): ...

    def _eval_straints(self, t, q, la_c, la_g, num_per_element): ...


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
        # TODO: change order
        # The order is the same!

    return BoostedCosseratRod_PetrovGalerkin
