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
    quatprod,
)
from cardillo.math.SmallestRotation import SmallestRotation
from cardillo.utility.coo_matrix import CooMatrix

from ._base_export import RodExportBase
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


class KirchhoffLoveRod_PetrovGalerkin(RodExportBase):
    def __init__(
        self,
        cross_section,
        material_model,
        cross_section_inertias,
        idx_displacement_based,
        idx_constraints,
        distributed_load,
        nelement,
        quadrature_int,
        quadrature_dyn,
        quadrature_ext,
        rotation_interpolation,
        full_inertia,
        Q,
        q0,
        u0,
        name,
    ):
        # TODO: can we also move this to the update parameters?
        # call base class for all export properties
        super().__init__(cross_section)

        self.orientation_type = rotation_interpolation

        # TODO: 4 or 6 strains?
        self.idx_g = idx_constraints
        self.idx_db = idx_displacement_based
        self.idx_c = np.setdiff1d(
            np.arange(6), np.union1d(idx_constraints, idx_displacement_based)
        )

        self.name = "Kirchhoff_rod" if name is None else name

        # discretization
        self.nelement = nelement
        self.nnodes = nelement + 1

        mesh_rP = Mesh1D_equidistant(
            basis="Hermite_C0",
            nelement=nelement,
            polynomial_degere=3,
            derivative_order=2,
        )
        mesh_alpha = Mesh1D_equidistant(
            basis="Lagrange",
            nelement=nelement,
            polynomial_degere=2,
            derivative_order=1,
        )
        # TODO: quadratic for n and linear for m I guess
        # mesh_cg = Mesh1D_equidistant(
        # basis=
        # )

        # element intervals
        self.element_interval = mesh_rP.element_interval
        self.element_number = mesh_rP.element_number
        self.node_number = mesh_rP.node_number

        self.h3 = lambda xis, els: mesh_rP.shape_functions(xis, els, 2)
        self.h3_element = lambda xi, el: mesh_rP.shape_function_array_element(xi, el, 0)

        self.N = lambda xis, els: mesh_alpha.shape_functions(xis, els, 1)
        self.N_element = lambda xi, el: mesh_alpha.shape_function_array_element(
            xi, el, 0
        )

        #####################
        # quadrature points #
        #####################
        # internal virtual work contribution
        quadrature_int_N = mesh_rP.quadrature(3, "Gauss", 2)
        self.nquadrature_int_total = quadrature_int_N["nquadrature_total"]
        self.qp_int_vec = quadrature_int_N["qp"]
        self.qw_int_vec = quadrature_int_N["qw"]
        self.qels_int_vec = quadrature_int_N["els"]
        self.h3_int, self.h3_xi_int, self.h3_xixi_int = quadrature_int_N["N"]

        quadrature_int_N = mesh_alpha.quadrature(3, "Gauss", 1)
        self.N_int, self.N_xi_int = quadrature_int_N["N"]

        self.shape_functions_int = (
            self.h3_int,
            self.h3_xi_int,
            self.h3_xixi_int,
            self.N_int,
            self.N_xi_int,
        )

        # make properties for nq, nu, nla_c, nla_g
        # and functions for g, c, and all related ones
        self._create_system_interfaces()

        # referential generalized position coordinates, initial generalized
        # position and velocity coordinates
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else u0

        # reference strains for weight of blocks
        self.set_reference_strains(Q)

        ################
        # assemble matrices for block structure
        ################
        # M
        M_pairs = ...

        # if self._nla_c > 0:
        #     # c_la_c
        #     c_la_c_pairs = ...

        # c_sigma
        c_sigma_q_pairs = ...

        # W_sigma
        W_sigma_pairs = ...

        # Wla_sigma_q
        h_pot_q_pairs = ...

        # set all parameters of the rod
        self.set_parameter(
            cross_section=cross_section,
            material_model=material_model,
            cross_section_inertias=cross_section_inertias,
            distributed_load=distributed_load,
        )

        # TODO: do inner dict with enum as key or use a class instead of dict
        self.interaction_points: dict[float, dict[str, np.ndarray]] = {}

    def _create_system_interfaces(self):
        # DOF handling
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

        # number of DB
        self._nDB = 6

        self.include_f_pot = True

    def set_parameter(
        self,
        *,
        cross_section=None,
        material_model=None,
        cross_section_inertias=None,
        distributed_load=None,
    ):
        if cross_section is not None:
            self.cross_section = cross_section
            if self.preprocessed_export:
                self.preprocess_export()

        if material_model is not None:
            self.material_model = material_model
            self.material_model.prepare_quadrature(self.qp_int_vec)

        if cross_section_inertias is not None:
            if cross_section_inertias == False:
                self.include_f_gyr = False
                self.cross_section_inertias = CrossSectionInertias_new()
            else:
                raise NotImplementedError
                # TODO: think of only using A_rho0 and B_I_rho0[0, 0]
                self.include_f_gyr = True
                self.cross_section_inertias = cross_section_inertias

            warn("Wrong quadrature for cross section inertias")
            self.cross_section_inertias.prepare_quadrature(self.qp_int_vec)
            # self.cross_section_inertias.prepare_quadrature(self.qp_dyn_vec)
        self.include_f_gyr = False

        if distributed_load is not None:
            assert (
                len(distributed_load) == 2
            ), "Line distributed forces must be a list of length 2 (force and moment)."
            if distributed_load[0] == None:
                self.include_f_ext = False
            else:
                self.distributed_load = distributed_load
                self.include_f_ext = True

            assert (
                distributed_load[1] == None
            ), "Line distributed moments are not allowed."

        # compose E_pot, h, h_q and h_u
        self.compose_E_h()

    def compose_E_h(self):
        # TODO: unify with other rod(s)?
        # compose h vector and potential energy
        # 1) collect contributions
        E_pot_functions = []
        h_functions = []
        h_q_functions = []
        h_u_functions = []

        # gyroscopic forces
        if self.include_f_gyr:
            h_functions.append(self.f_gyr)
            h_u_functions.append(self.f_gyr_u)

        # displacement based potential forces
        if self.include_f_pot:
            E_pot_functions.append(self.E_pot_int)
            h_functions.append(self.f_pot)
            h_q_functions.append(self.f_pot_q)

        # line distributed forces
        if self.include_f_ext:
            E_pot_functions.append(self.E_pot_ext)
            h_functions.append(self.f_ext)

        # 2) add them up
        if len(E_pot_functions) > 0:
            self.E_pot = lambda t, q: np.sum(
                [Ei(t, q) for Ei in E_pot_functions], axis=0
            )
        elif hasattr(self, "E_pot"):
            delattr(self, "E_pot")

        if len(h_functions) > 0:
            self.h = lambda t, q, u: np.sum([hi(t, q, u) for hi in h_functions], axis=0)
        elif hasattr(self, "h"):
            delattr(self, "h")

        if len(h_q_functions) > 0:
            self.h_q = lambda t, q, u: np.sum(
                [hi_q(t, q, u) for hi_q in h_q_functions], axis=0
            )
        elif hasattr(self, "h_q"):
            delattr(self, "h_q")

        if len(h_u_functions) > 0:
            self.h_u = lambda t, q, u: np.sum(
                [hi_u(t, q, u) for hi_u in h_u_functions], axis=0
            )
        elif hasattr(self, "h_u"):
            delattr(self, "h_u")

    @staticmethod
    def straight_configuration(
        nelement,
        L,
        r_OP0=np.zeros(3, dtype=float),
        A_IB0=np.eye(3, dtype=float),
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
            # TODO: remove r_OC
            return r_OC, A_IB, j, B_kappa_bar

        if choice == "int":
            return r_OC, A_IB, j, B_kappa_bar, ex_B_xi, ex_B_to_j, ex_B_to_j_xi

        print("Not implemented yet")

    def set_reference_strains(self, Q):
        self.Q = Q.copy()

        _, _, J, B_kappa0_bar = self._eval_int_vec(
            self.shape_functions_int, self.Q, choice="strains"
        )
        self.J_int_vec = J
        self.B_Gamma0_int = np.zeros((self.nquadrature_int_total, 3), dtype=float)
        self.B_Gamma0_int[:, 0] = 1.0  # J / J = 1.0
        self.B_kappa0_bar_int = B_kappa0_bar
        self.B_Kappa0_int = self.B_kappa0_bar_int / J[:, None]
        self.epsilon0_int = np.hstack([self.B_Gamma0_int, self.B_Kappa0_int])
        self._epsilon_int = np.zeros((self.nquadrature_int_total, 6))

        # TODO: dyn and ext?

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
    def elDOF_P(self, xi):
        # TODO: do this function seriously. Can we even get rid of it?
        # TODO: move to rod parent
        return self.get_interaction_point(xi).get("qDOF")

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

    def local_qDOF_P(self, xi):
        # TODO: move to rod parent
        return self.get_interaction_point(xi).get("qDOF")

    def local_uDOF_P(self, xi):
        # TODO: move to rod parent
        return self.get_interaction_point(xi).get("uDOF")

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
        if False:
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
        warn("Not implemented yet")
        return 0.0

    def f_ext(self, t, q, u):
        warn("Not implemented yet")
        return np.zeros(self.nu)

    #########################
    # internal virtual work #
    #########################
    def l_sigma(self, q): ...
    def l_sigma_q(self, q): ...
    def _c_la_c_coo(self): ...
    def W_sigma(self, q): ...
    def Wla_sigma(self, q, la_c=None, la_g=None): ...
    def h_pot(self, _eval, sigma_qp): ...

    def f_pot(self, t, q, u):
        _, A_IB, j, B_kappa_bar, ex_B_xi, ex_B_to_j, ex_B_to_j_xi = self._eval_int_vec(
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

        #######
        # TODO: handle from here on in a sperate function
        #######
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
        return -np.concatenate(
            [h_rt[: self.nnodes].reshape(-1), h_psi.reshape(-1), h_j_pm, h_alpha]
        )

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
            _, _, J, B_kappa0_bar = self._eval_int_vec(
                shape_functions, self.Q, choice="strains"
            )
            # current strains
            _, _, j, B_kappa_bar = self._eval_int_vec(
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
    # TODO: think of counting 0-5 or only 0-3
    # constraints
    assert idx_constraints is None
    idx_constraints = np.array([], dtype=int)

    # displacement based
    assert idx_displacement_based is None
    idx_displacement_based = np.array([], dtype=int)

    # quadrature
    if quadrature_int == None:
        quadrature_int = (3, "Gauss")
    elif isinstance(quadrature_int, int):
        quadrature_int = (quadrature_int, "Gauss")
    elif not isinstance(quadrature_int, tuple):
        raise ValueError(
            "quadrature_int must be either an 'None', an integer for Gauss quadrature or a tuple: (nquadrature, method)"
        )

    # # assert quadrature_dyn is None
    # if quadrature_dyn == None:
    #     # TODO: take trapezoidal rule as default?
    #     quadrature_dyn = (polynomial_degree + 1, "Trapezoidal")
    #     n_full = int(np.ceil(3 / 2 * polynomial_degree + 1 / 2))
    #     quadrature_dyn = (n_full, "Gauss")
    #     print(f"quadrature_dyn: {quadrature_dyn}")
    # elif isinstance(quadrature_dyn, int):
    #     quadrature_dyn = (quadrature_dyn, "Gauss")
    # elif not isinstance(quadrature_dyn, tuple):
    #     raise ValueError(
    #         "quadrature_dyn must be either an 'None', an integer for Gauss quadrature or a tuple: (nquadrature, method)"
    #     )

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
        def __init__(
            self,
            cross_section,
            material_model,
            nelement,
            *,
            Q,
            q0=None,
            u0=None,
            distributed_load=[None, None],
            cross_section_inertias=False,
            name="KirchhoffLove_rod",
        ):
            super().__init__(
                cross_section,
                material_model,
                cross_section_inertias,
                idx_constraints,
                idx_displacement_based,
                distributed_load,
                nelement,
                quadrature_int,
                quadrature_dyn,
                quadrature_ext,
                rotation_interpolation,
                full_inertia,
                Q,
                q0,
                u0,
                name,
            )

    return KirchhoffLoveRod
