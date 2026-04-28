from abc import ABC, abstractmethod
import numpy as np

from ._cross_section import CrossSectionInertias_new
from ._base_export import RodExportBase
from cardillo.utility.check_time_derivatives import check_time_derivatives

zeros3 = np.zeros(3, dtype=float)
eye3 = np.eye(3, dtype=float)


class RodInterface(RodExportBase):
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
        """Cardillo rod formulation

        Parameters
        ----------
        cross_section : CrossSection
            Geometric cross-section properties: area, first and second moments of area.
        material_model: RodMaterialModel
            Constitutive law of Cosserat rod which relates the rod strain
            measures B_gamma and B_kappa with the contact forces B_n and couples
            B_m in the cross-section-fixed B-basis.
        nelement : int
            Number of rod elements.
        Q : np.ndarray (self.nq,)
            Generalized position coordinates of rod in a stress-free reference
            state. Q is a collection of nodal generalized position coordinates.
        q0 : np.ndarray (self.nq,)
            Initial generalized position coordinates of rod at time t0.
        u0 : np.ndarray (self.nu,)
            Initial generalized velocity coordinates of rod at time t0.
        distributed_load : list (2,)
            distributed_load[0] : I_b(t, xis) callable function for distributed force or None
            distributed_load[1] : B_c(t, xis) callable function for distributed moment or None
        cross_section_inertias : CrossSectionInertias
            Inertia properties of cross-sections: Cross-section mass density and
            Cross-section inertia tensor represented in the cross-section-fixed
            B-Basis.
        name : str
            Name of contribution.
        """
        self._pre_init_()

        super().__init__(cross_section)
        self.name = name
        self.nelement = nelement

        # create FEM mesh
        self._create_meshs()

        # set nq, nu, nla_c, nla_g
        self._create_system_interfaces()

        # referential generalized position coordinates, initial generalized
        # position and velocity coordinates
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else u0

        # reference strains
        self._set_reference_strains(Q)

        # set all parameters of the rod
        self.set_parameter(
            cross_section=cross_section,
            material_model=material_model,
            cross_section_inertias=cross_section_inertias,
            distributed_load=distributed_load,
        )

        # TODO: do inner dict with enum as key or use a class instead of dict
        self.interaction_points: dict[float, dict[str, np.ndarray]] = {}

        # create
        # self._create_SparseArrayBlocks()
        self._post_init_()

    ########################
    # sub methods for init #
    ########################
    @abstractmethod
    def _pre_init_(self): ...
    @abstractmethod
    def _create_meshs(self): ...
    @abstractmethod
    def _create_system_interfaces(self): ...
    @abstractmethod
    def _set_reference_strains(self, Q): ...
    @abstractmethod
    def _post_init_(self): ...

    def _handle_internal(self, nla_c, nla_g):
        # compliance
        if nla_c > 0:
            self.nla_c = self._nla_c = nla_c

            ##############
            # compliance #
            # c = c_la_c @ la_c - l_c
            ##############
            # la_c = -c_la_c_inv @ c(q, u, 0) = c_la_c_inv @ l_c(q)
            self.la_c = lambda t, q, u: self._cla_c_inv @ self.l_sigma(q)[0]
            self.c = lambda t, q, u, la_c: self._cla_c @ la_c - self.l_sigma(q)[0]
            self.c_q = lambda t, q, u, la_c: -self.l_sigma_q(q)[0]
            self.c_la_c = lambda: self._cla_c
            self.W_c = lambda t, q: self.W_sigma(q)[0]
            self.Wla_c_q = lambda t, q, la_c: self.Wla_sigma_q(q, la_c, None)
            self.E_pot_comp = self._E_pot_comp
        else:
            self._nla_c = 0

        # constraint
        # director beam passes 0 here
        if nla_g > 0:
            self.nla_g = self._nla_g = nla_g

            #############
            # constraints
            #  g = - l_g
            #############
            self.g = lambda t, q: -self.l_sigma(q)[1]
            self.g_q = lambda t, q: -self.l_sigma_q(q)[1]
            self.W_g = lambda t, q: self.W_sigma(q)[1]
            self.Wla_g_q = lambda t, q, la_g: self.Wla_sigma_q(q, None, la_g)

            self.g_dot = lambda t, q, u: self.W_sigma(q)[1].T @ u
            self.g_dot_u = lambda t, q: self.W_sigma(q)[1].T
            self.g_dot_q = lambda t, q, u: self.l_sigma_dot_q(q, u)[1]
            self.g_ddot = lambda t, q, u, u_dot: self.l_sigma_ddot(q, u, u_dot)[1]

        else:
            self._nla_g = 0

        # displacement-based
        self._nDB = len(self.idx_db)
        self.include_f_pot = self._nDB > 0

    # methods to change
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
                self.include_f_gyr = True
                self.cross_section_inertias = cross_section_inertias

            self.cross_section_inertias.prepare_quadrature(self.qp_dyn_vec)

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

        # gyroscopic forces
        if self.include_f_gyr:
            if hasattr(self, "f_gyr"):
                h_functions.append(self.f_gyr)
            if hasattr(self, "f_gyr_u"):
                h_u_functions.append(self.f_gyr_u)

        # displacement based potential forces
        if self.include_f_pot:
            if hasattr(self, "E_pot_int"):
                E_pot_functions.append(self.E_pot_int)
            if hasattr(self, "f_pot"):
                h_functions.append(self.f_pot)
            if hasattr(self, "f_pot_q"):
                h_q_functions.append(self.f_pot_q)
            if hasattr(self, "f_pot_u"):
                h_u_functions.append(self.f_pot_u)

        # line distributed forces
        if self.include_f_ext:
            if hasattr(self, "E_pot_ext"):
                E_pot_functions.append(self.E_pot_ext)
            if hasattr(self, "f_ext"):
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

    ####################################################
    # interactions with other bodies and the environment
    ####################################################
    def local_qDOF_P(self, xi):
        return self.get_interaction_point(xi).get("qDOF")

    def local_uDOF_P(self, xi):
        return self.get_interaction_point(xi).get("uDOF")

    @abstractmethod
    def get_interaction_point(self, xi): ...

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

    @abstractmethod
    def eval_stresses(self, t, q, la_c, la_g, n_per_element=None, n_ges=None): ...
    @abstractmethod
    def eval_strains(self, t, q, la_c, la_g, n_per_element=None, n_ges=None): ...

    ##################
    # configurations #
    ##################
    @staticmethod
    @abstractmethod
    def straight_configuration(nelement, L, r_OP0=zeros3, A_IB0=eye3):
        """Compute generalized position coordinates for straight configuration."""
        ...

    @staticmethod
    @abstractmethod
    def pose_configuration(nelement, r_OP, A_IB, xi1=1.0, r_OP0=zeros3, A_IB0=eye3):
        """Compute generalized position coordinates for a pre-curved rod with centerline curve r_OP and orientation of A_IB."""
        ...

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

    @staticmethod
    @abstractmethod
    def straight_initial_configuration(
        nelement, L, r_OP0=zeros3, A_IB0=eye3, v_P0=zeros3, B_omega_IB0=eye3
    ): ...
