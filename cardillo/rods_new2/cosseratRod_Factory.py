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
from warnings import warn

from cardillo.math.algebra import norm, cross3, ax2skew, ax2skew_a
from cardillo.math.approx_fprime import approx_fprime

from cardillo.utility.coo_matrix import CooMatrix
from cardillo.utility.sparse_array_blocks import SparseArrayBlocks

from cardillo.rods._base_export import RodExportBase


from cardillo.rods_new.discretization.mesh1D import Mesh1D_equidistant, Mesh1D_IGA

from cardillo.rods_new2.cosseratRod_Kinematics import (
    Rod_Kinematics,
    CosseratRod_Quaternion_R12,
)
from cardillo.rods_new2.cosseratRod_Velocity import Rod_Velocity, CosseratRod_PG_IB
from cardillo.rods_new2.cosseratRod_Interaction import CosseratRod_Interaction
from cardillo.rods_new2.cosseratRod_q_dot import (
    CosseratRod_rP_dot_from_vO_IB,
    CosseratRod_kin_trivial,
)

from cardillo.rods_new2.cosseratRod_Internal import CosseratRod_internal_PG_IB
from cardillo.rods_new2.cosseratRod_Dynamics import CosseratRod_dynamics_PG_IB


class CosseratRod:
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
        name="Rod",
    ):
        # TODO: export
        self.preprocessed_export = False

        self.name = name
        self.nelement = nelement

        # safe evaluation for r_OP, A_IB, ....
        self.interaction_points: dict[float, dict[str, np.ndarray]] = {}

        ###################
        # create FEM mesh #
        ###################
        mesh_kin = self._mesh_kin(nelement)
        mesh_cg = self._mesh_cg(nelement)

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
        # external virtual work contributions
        quadrature_ext = mesh_kin.quadrature(*self._quadrature_ext, 1)
        self.nquadrature_ext_total = quadrature_ext["nquadrature_total"]
        self.qp_ext_vec = quadrature_ext["qp"]
        self.qw_ext_vec = quadrature_ext["qw"]
        self.qels_ext_vec = quadrature_ext["els"]
        self.N_ext, self.N_xi_ext = quadrature_ext["N"]

        # initialize position/orientation interpolation
        self.kinematics = self._Kinematics(self, mesh_kin, self._parametrization)

        # initialize velocities interpolation
        self.velocity = self._Velocity(self, mesh_kin)

        ############################
        # create system interfaces #
        ############################
        # total number of generalized position and velocity coordinates
        self.nq = self.nnodes * self.kinematics.nq_node
        self.nu = self.nnodes * self.velocity.nu_node

        # initialize kinematics equation
        self.kin_eq = self._Kin_eq(self, self._parametrization, self._projection)

        # initialize internal virtual work contributions
        self.internal = self._Internal(self, mesh_kin, mesh_cg, self._quadrature_int)

        # initialize dynamics virtual work contributiosn
        self.dynamics = self._Dynamics(
            self, mesh_kin, self._quadrature_dyn, self._projection
        )

        # handle g/g_S
        self.compose_g()

        ##########################################
        # reference configuration                #
        # init. generalized position coordinates #
        # init. generalized velocity coordinates #
        ##########################################
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else u0

        # reference strains
        self.set_reference_strains(Q)

        # set all parameters of the rod
        self.set_parameter(
            cross_section=cross_section,
            material_model=material_model,
            cross_section_inertias=cross_section_inertias,
            distributed_load=distributed_load,
        )

        # TODO: post init, i.e., creating the matrix blocks

    def set_reference_strains(self, Q):
        self.Q = Q.copy()

        self.internal.set_reference_strains(Q)
        self.dynamics.set_reference_strains(Q)

        # external virtual work contributions
        _, B_gamma_bar, _ = self.kinematics._eval_internal_vec(
            self.N_ext, self.N_xi_ext, self.Q
        )
        J_ext_vec = np.linalg.norm(B_gamma_bar, axis=1)
        self.weights_ext = J_ext_vec * self.qw_ext_vec

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
            self.internal.set_material_model(material_model)

        if cross_section_inertias is not None:
            self.dynamics.set_cross_section_inertias(cross_section_inertias)

        if distributed_load is not None:
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

        # compose E_pot, h, h_q and h_u
        self.compose_E_h()

    def compose_E_h(self):
        # compose h vector and potential energy
        # 1) collect contributions
        E_pot_functions = []
        h_functions = []
        h_q_functions = []
        h_u_functions = []
        # TODO: KN_h?

        # gyroscopic forces
        if self.dynamics.include_f_gyr:
            if hasattr(self.dynamics, "f_gyr"):
                h_functions.append(self.dynamics.f_gyr)
            if hasattr(self.dynamics, "f_gyr_q"):
                h_q_functions.append(self.dynamics.f_gyr_q)
            if hasattr(self.dynamics, "f_gyr_u"):
                h_u_functions.append(self.dynamics.f_gyr_u)

        # displacement based potential forces
        if self.internal.include_f_pot:
            if hasattr(self.internal, "E_pot"):
                E_pot_functions.append(self.internal.E_pot)
            if hasattr(self.internal, "f_pot"):
                h_functions.append(self.internal.f_pot)
            if hasattr(self.internal, "f_pot_q"):
                h_q_functions.append(self.internal.f_pot_q)
            if hasattr(self.internal, "f_pot_u"):
                h_u_functions.append(self.internal.f_pot_u)

        # line distributed forces
        if self.include_f_ext:
            # if hasattr(self, "E_pot_ext"):
            E_pot_functions.append(self.E_pot_ext)
            # if hasattr(self, "f_ext"):
            h_functions.append(self.f_ext)

        # 2) add them up
        if len(E_pot_functions) == 1:
            self.E_pot = E_pot_functions[0]
        elif len(E_pot_functions) > 1:
            self.E_pot = lambda t, q: np.sum(
                [Ei(t, q) for Ei in E_pot_functions], axis=0
            )
        elif hasattr(self, "E_pot"):
            delattr(self, "E_pot")

        if len(h_functions) == 1:
            self.h = h_functions[0]
        elif len(h_functions) > 1:
            self.h = lambda t, q, u: np.sum([hi(t, q, u) for hi in h_functions], axis=0)
        elif hasattr(self, "h"):
            delattr(self, "h")

        if len(h_q_functions) == 1:
            self.h_q = h_q_functions[0]
        elif len(h_q_functions) > 1:
            self.h_q = lambda t, q, u: np.sum(
                [hi_q(t, q, u) for hi_q in h_q_functions], axis=0
            )
        elif hasattr(self, "h_q"):
            delattr(self, "h_q")

        if len(h_u_functions) == 1:
            self.h_u = h_u_functions[0]
        elif len(h_u_functions) > 1:
            self.h_u = lambda t, q, u: np.sum(
                [hi_u(t, q, u) for hi_u in h_u_functions], axis=0
            )
        elif hasattr(self, "h_u"):
            delattr(self, "h_u")

    def compose_g(self):
        # 1) collect functions
        g_functions = []
        g_q_functions = []
        W_g_functions = []
        Wla_g_q_functions = []
        KN_g_functions = []

        g_dot_functions = []
        g_dot_u_functions = []
        g_dot_q_functions = []
        g_ddot_functions = []

        # deformation constraints
        _nla_g = 0
        if self.internal.include_g:
            la_gDOF_int = slice(_nla_g, _nla_g + self.internal._nla_g)
            _nla_g += self.internal._nla_g
            g_functions.append(lambda t, q: -self.internal.l_sigma(q)[1])
            g_q_functions.append(lambda t, q: -self.internal.l_sigma_q(q)[1])
            W_g_functions.append(lambda t, q: self.internal.W_sigma(q)[1])
            Wla_g_q_functions.append(
                lambda t, q, la_g: self.internal.Wla_sigma_q(q, None, la_g[la_gDOF_int])
            )
            KN_g_functions.append(
                lambda t, q, la_g: (
                    self.internal.K_sigma(q, None, la_g[la_gDOF_int]),
                    None,
                )
            )

            g_dot_functions.append(lambda t, q, u: self.internal.W_sigma(q)[1].T @ u)
            g_dot_u_functions.append(lambda t, q: self.internal.W_sigma(q)[1].T)
            g_dot_q_functions.append(
                lambda t, q, u: self.internal.l_sigma_dot_q(q, u)[1]
            )  # TODO: why no minus?
            g_ddot_functions.append(
                lambda t, q, u, u_dot: self.internal.l_sigma_ddot(q, u, u_dot)[1]
            )  # TODO: why no minus?

        if self.kin_eq.include_g:
            la_gDOF_kin = slice(_nla_g, _nla_g + self.kin_eq._nla_g)
            _nla_g += self.kin_eq._nla_g
            g_functions.append(self.kin_eq.g)
            g_q_functions.append(self.kin_eq.g_q)
            W_g_functions.append(self.kin_eq.W_g)
            Wla_g_q_functions.append(
                lambda t, q, la_g: self.kin_eq.Wla_g_q(t, q, la_g[la_gDOF_kin])
            )
            KN_g_functions.append(
                lambda t, q, la_g: self.kin_eq.KN_g(t, q, la_g[la_gDOF_kin])
            )

            g_dot_functions.append(self.kin_eq.g_dot)
            g_dot_u_functions.append(self.kin_eq.g_dot_u)
            g_dot_q_functions.append(self.kin_eq.g_dot_q)
            g_ddot_functions.append(self.kin_eq.g_ddot)

        if self.kin_eq.include_g_S:
            self.nla_S = self.kin_eq._nla_g
            self.g_S = self.kin_eq.g
            self.g_S_q = self.kin_eq.g_q

        # 2) compose
        if _nla_g > 0:
            self.nla_g = _nla_g
        if len(g_functions) == 1:
            self.g = g_functions[0]
        elif len(g_functions) > 1:
            self.g = lambda t, q: np.concatenate([g(t, q) for g in g_functions])

        if len(g_q_functions) == 1:
            self.g_q = g_q_functions[0]
        elif len(g_q_functions) > 1:
            self.g_q = lambda t, q: np.concatenate([g_q(t, q) for g_q in g_q_functions])

        if len(W_g_functions) == 1:
            self.W_g = W_g_functions[0]
        elif len(W_g_functions) > 1:
            self.W_g = lambda t, q: np.concatenate(
                [W_g(t, q) for W_g in W_g_functions], axis=1
            )

        if len(Wla_g_q_functions) == 1:
            self.Wla_g_q = Wla_g_q_functions[0]
        elif len(Wla_g_q_functions) > 1:
            self.Wla_g_q = lambda t, q, la_g: np.sum(
                [Wla_g_q(t, q, la_g) for Wla_g_q in Wla_g_q_functions], axis=0
            )

        if len(KN_g_functions) == 1:
            self.KN_g = KN_g_functions[0]
        elif len(KN_g_functions) > 1:
            self.KN_g = lambda t, q, la_g: np.sum(
                [KN_g(t, q, la_g) for KN_g in KN_g_functions], axis=0
            )

        if len(g_dot_functions) == 1:
            self.g_dot = g_dot_functions[0]
        elif len(g_dot_functions) > 1:
            self.g_dot = lambda t, q, u: np.concatenate(
                [g_dot(t, q, u) for g_dot in g_dot_functions]
            )

        if len(g_dot_u_functions) == 1:
            self.g_dot_u = g_dot_u_functions[0]
        elif len(g_dot_u_functions) > 1:
            self.g_dot_u = lambda t, q: np.concatenate(
                [g_dot_u(t, q) for g_dot_u in g_dot_u_functions]
            )

        if len(g_dot_q_functions) == 1:
            self.g_dot_q = g_dot_q_functions[0]
        elif len(g_dot_q_functions) > 1:
            self.g_dot_q = lambda t, q, u: np.concatenate(
                [g_dot_q(t, q, u) for g_dot_q in g_dot_q_functions]
            )

        if len(g_ddot_functions) == 1:
            self.g_ddot = g_ddot_functions[0]
        elif len(g_ddot_functions) > 1:
            self.g_ddot = lambda t, q, u, u_dot: np.concatenate(
                [g_ddot(t, q, u, u_dot) for g_ddot in g_ddot_functions]
            )

    def assembler_callback(self):
        if hasattr(self.dynamics, "assembler_callback"):
            self.dynamics.assembler_callback()

        if hasattr(self.internal, "assembler_callback"):
            self.internal.assembler_callback()


def make_CosseratRod(
    *,
    polynomial_degree=None,
    continuity=None,
    idx_constraints=None,
    idx_displacement_based=None,
    quadrature_int=None,
    quadrature_dyn=None,
    quadrature_ext=None,
    parametrization=None,
):
    """Factory for Petrov-Galerkin Cosserat rod classes.

    Parameters
    ----------
    polynomial_degree : int, optional
        Polynomial degree (p) of the interpolation of centerline, orientation,
        virtual displacement, and virtual rotation. If not specified, p = 2 is used.

    continuity : int, optional
        If None: C^0 Lagrange elements, otherwise C^(continuity) B-spline elements

    idx_constraints : array_like of int
        Indices (0-5) of constrained strain components. Must not overlap with "idx_displacement_based".

    idx_displacement_based : array_like of int
        Indices (0-5) of displacement-based strain components. Must not overlap with "idx_constraints".

    quadrature_int : int or tuple[int, str]
        Quadrature rule for internal virtual work integration.

    quadrature_dyn : int or tuple[int, str]
        Quadrature rule for dynamic virtual work integration.

    quadrature_ext : int or tuple[int, str]
        Quadrature rule for external virtual work integration.

    parametrization : str
        Choice of parametrization and interpolation


    Strain component mapping
    ----------
        0 : Gamma_1 (dilatation)
        1 : Gamma_2 (shear in e_y^B direction)
        2 : Gamma_3 (shear in e_z^B direction)
        3 : kappa_1 (torsion)
        4 : kappa_2 (bending around e_y^B)
        5 : kappa_3 (bending around e_z^B)


    Parametrization
    ----------
        - "Quaternion" : quaternion parametrization and interpolation
        - "R12" : R12 parametrization and interpolation

    Returns
    -------
    CosseratRod
        Constructed rod class.
    """
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

    if continuity is None:
        mesh_kin = lambda _, nelement: Mesh1D_equidistant(
            "Lagrange", nelement, polynomial_degree, 1
        )
        mesh_cg = lambda _, nelement: Mesh1D_equidistant(
            "Lagrange_Disc", nelement, polynomial_degree - 1, 0
        )
    else:
        mesh_kin = lambda _, nelement: Mesh1D_IGA(
            nelement, polynomial_degree, continuity, 1
        )
        mesh_cg = lambda _, nelement: Mesh1D_IGA(
            nelement, polynomial_degree - 1, np.max(continuity - 1, -1), 0
        )

    # TODO: select these properly!
    Kinematics = CosseratRod_Quaternion_R12
    Velocity = CosseratRod_PG_IB
    Kin_eq = CosseratRod_rP_dot_from_vO_IB
    Internal = CosseratRod_internal_PG_IB
    Dynamics = CosseratRod_dynamics_PG_IB

    class _CosseratRod(
        CosseratRod, Rod_Kinematics, Rod_Velocity, CosseratRod_Interaction
    ):
        _polynomial_degree = polynomial_degree
        _mesh_kin = mesh_kin
        _mesh_cg = mesh_cg
        _Kinematics = Kinematics
        _Velocity = Velocity
        _Kin_eq = Kin_eq
        _Internal = Internal
        _Dynamics = Dynamics
        _parametrization = parametrization
        _projection = True
        _IGA = not (continuity is None)

        _quadrature_int = quadrature_int
        _quadrature_dyn = quadrature_dyn
        _quadrature_ext = quadrature_ext

        idx_g = idx_constraints
        idx_db = idx_displacement_based
        idx_c = np.setdiff1d(
            np.arange(6), np.union1d(idx_constraints, idx_displacement_based)
        )

    return _CosseratRod


if __name__ == "__main__":
    cs = None
    mm = None
    nelement = 8
    nnodes = 2 * nelement + 1
    Q = np.random.rand(7 * nnodes)

    Rod = make_CosseratRod()
    Q = Rod.straight_configuration(nelement, 5)
    rod = Rod(cs, mm, nelement, Q=Q)

    rod.r_OP
