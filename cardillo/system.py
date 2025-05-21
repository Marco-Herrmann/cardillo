import numpy as np
import warnings
from copy import deepcopy
import scipy

from cardillo.utility.coo_matrix import CooMatrix
from cardillo.discrete.frame import Frame
from cardillo.discrete.meshed import Axis
from cardillo.math.approx_fprime import approx_fprime
from cardillo.solver import consistent_initial_conditions, Solution
from cardillo.visualization import Export

properties = []

properties.extend(["E_kin", "E_pot"])

properties.extend(["M", "Mu_q"])

properties.extend(["h", "h_q", "h_u"])

properties.extend(["q_dot", "q_dot_q", "q_dot_u"])

properties.extend(["g"])
properties.extend(["gamma"])

properties.extend(["c", "c_q", "c_u"])
properties.extend(["g_S"])

properties.extend(["la_tau"])
properties.extend(["tau"])

properties.extend(["g_N"])
properties.extend(["gamma_F", "gamma_F_q"])

properties.extend(["assembler_callback", "step_callback"])


class System:
    """Sparse model implementation which assembles all global objects without
    copying on body and element level.

    Parameters
    ----------
    t0 : float
        Initial time of the initial state of the system.
    origin_size: float
        Origin size for trimesh visualization.
        If origin_size>0, the origin of the system is added as trimesh.axis with the specified origin size. Otherwise the system origin is just a cardillo Frame.

    Notes
    -----

    All model functions which return matrices have :py:class:`scipy.sparse.coo_array`
    as default scipy sparse matrix type (:py:class:`scipy.sparse.spmatrix`).
    This is due to the fact that the assembling of global iteration matrices
    is done using :py:func:`scipy.sparse.bmat` which in a first step transforms
    all matrices to :py:class:`scipy.sparse.coo_array`. A :py:class:`scipy.sparse.coo_array`,
    inherits form :py:class:`scipy.sparse._data_matrix`
    `[1] <https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/data.py#L21-L126>`_,
    have limited support for arithmetic operations, only a few operations as
    :py:func:`__neg__`, :py:func:`__imul__`, :py:func:`__itruediv__` are implemented.
    For all other operations the matrix is first transformed to a :py:class:`scipy.sparse.csr_array`
    `[2] <https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/base.py#L330-L335>`_.
    Slicing is also not supported for matrices of type :py:class:`scipy.sparse.coo_array`,
    we have to use other formats as :py:class:`scipy.sparse.csr_array` or
    :py:class:`scipy.sparse.csc_array` for that.

    """

    def __init__(self, t0=0, origin_size=0):
        self.t0 = t0
        self.nq = 0
        self.nu = 0
        self.nla_g = 0
        self.nla_gamma = 0
        self.nla_c = 0
        self.nla_tau = 0
        self.ntau = 0
        self.nla_S = 0
        self.nla_N = 0
        self.nla_F = 0

        self.contributions = []
        self.contributions_map = {}
        self.ncontr = 0

        if origin_size > 0:
            self.origin = Axis(Frame)(origin_size=origin_size)
        else:
            self.origin = Frame()

        self.origin.name = "cardillo_origin"
        self.add(self.origin)

    def add(self, *contrs):
        """Adds contributions to the system.

        Parameters
        ----------
        contrs : object or list
            Single object or list of objects to add to the system.
        """
        for contr in contrs:
            if not contr in self.contributions:
                self.contributions.append(contr)
                if not hasattr(contr, "name"):
                    contr.name = "contr" + str(self.ncontr)

                if contr.name in self.contributions_map:
                    new_name = contr.name + "_contr" + str(self.ncontr)
                    print(
                        f"There is another contribution named '{contr.name}' which is already part of the system. Changed the name to '{new_name}' and added it to the system."
                    )
                    contr.name = new_name
                self.contributions_map[contr.name] = contr
                self.ncontr += 1
            else:
                raise ValueError(f"contribution {str(contr)} already added")

    def remove(self, *contrs):
        for contr in contrs:
            if contr in self.contributions:
                self.contributions.remove(contr)
            else:
                raise ValueError(f"no contribution {str(contr)} to remove")

    def pop(self, index):
        self.contributions.pop(index)

    def extend(self, contr_list):
        list(map(self.add, contr_list))

    def deepcopy(self):
        """
        Create a deepcopy of the system.

        Returns:
        --------
        system: cardillo.System
            deepcopy of the system
        """
        return deepcopy(self)

    def set_new_initial_state(self, q0, u0, t0=None, **assemble_kwargs):
        """
        Sets the initial state of the system.

        Parameters:
        -----------
        q0 : np.ndarray
            initial position coordinates
        u0 : np.ndarray
            initial velocity coordinates
        t0 : float
            initial time

        """
        self.t0 = t0 if t0 is not None else self.t0

        # extract final generalized coordiantes and distribute to subsystems
        for contr in self.contributions:
            if hasattr(contr, "nq"):
                contr.q0 = q0[contr.my_qDOF]

        # optionally distribute all other solution fields
        for contr in self.contributions:
            if hasattr(contr, "nu"):
                contr.u0 = u0[contr.my_uDOF]

        self.assemble(**assemble_kwargs)

    def export(self, path, folder_name, solution, overwrite=True, fps=50):
        e = Export(path, folder_name, overwrite, fps, solution)
        for contr in self.contributions:
            if hasattr(contr, "export"):
                e.export_contr(contr, file_name=contr.name)
        return e

    def get_contribution_list(self, contr):
        return getattr(self, f"_{self.__class__.__name__}__{contr}_contr")

    def reset(self):
        for contr in self.contributions:
            if hasattr(contr, "reset"):
                contr.reset()

    def assemble(self, *args, **kwargs):
        """Assembles the system, i.e., counts degrees of freedom, sets connectivities and assembles global initial state.

        Parameters
        ----------
        slice_active_contacts : bool
            When computing consistent initial conditions, slice friction forces to contemplate only those corresponding to active normal contact.
        options : cardillo.solver.SolverOptions
            Solver options for the computation of the constraint/contact forces.
        """
        self.nq = 0
        self.nu = 0
        self.nla_g = 0
        self.nla_gamma = 0
        self.nla_c = 0
        self.nla_tau = 0
        self.ntau = 0
        self.nla_S = 0
        self.nla_N = 0
        self.nla_F = 0
        q0 = []
        u0 = []
        e_N = []
        e_F = []
        self.constant_force_reservoir = False

        for p in properties:
            setattr(self, f"_{self.__class__.__name__}__{p}_contr", [])

        for contr in self.contributions:
            contr.t0 = self.t0
            for p in properties:
                # if property is implemented as class function append to property contribution
                # - p in contr.__class__.__dict__: has global class attribute p
                # - callable(getattr(contr, p, None)): p is callable
                if hasattr(contr, p) and callable(getattr(contr, p)):
                    getattr(self, f"_{self.__class__.__name__}__{p}_contr").append(
                        contr
                    )

            # if contribution has position degrees of freedom address position coordinates
            if hasattr(contr, "nq"):
                contr.my_qDOF = np.arange(0, contr.nq) + self.nq
                contr.qDOF = contr.my_qDOF.copy()
                self.nq += contr.nq
                q0.extend(contr.q0.tolist())

            # if contribution has velocity degrees of freedom address velocity coordinates
            if hasattr(contr, "nu"):
                contr.my_uDOF = np.arange(0, contr.nu) + self.nu
                contr.uDOF = contr.my_uDOF.copy()
                self.nu += contr.nu
                u0.extend(contr.u0.tolist())

            # if contribution has compliance contribution
            if hasattr(contr, "nla_c"):
                contr.la_cDOF = np.arange(0, contr.nla_c) + self.nla_c
                self.nla_c += contr.nla_c

            # if contribution of actuator forces
            if hasattr(contr, "nla_tau"):
                contr.la_tauDOF = np.arange(0, contr.nla_tau) + self.nla_tau
                self.nla_tau += contr.nla_tau

            # if contribution of control inputs
            if hasattr(contr, "ntau"):
                contr.tauDOF = np.arange(0, contr.ntau) + self.ntau
                self.ntau += contr.ntau

            # if contribution has constraints on position level address constraint coordinates
            if hasattr(contr, "nla_g"):
                contr.la_gDOF = np.arange(0, contr.nla_g) + self.nla_g
                self.nla_g += contr.nla_g

            # if contribution has constraints on velocity level address constraint coordinates
            if hasattr(contr, "nla_gamma"):
                contr.la_gammaDOF = np.arange(0, contr.nla_gamma) + self.nla_gamma
                self.nla_gamma += contr.nla_gamma

            # if contribution has stabilization conditions for the kinematic equation
            if hasattr(contr, "nla_S"):
                contr.la_SDOF = np.arange(0, contr.nla_S) + self.nla_S
                self.nla_S += contr.nla_S

            # if contribution has contact
            if hasattr(contr, "nla_N"):
                contr.la_NDOF = np.arange(0, contr.nla_N) + self.nla_N
                self.nla_N += contr.nla_N
                e_N.extend(contr.e_N.tolist())

            # if contribution has friction
            if hasattr(contr, "nla_F"):
                contr.la_FDOF = np.arange(0, contr.nla_F) + self.nla_F
                self.nla_F += contr.nla_F
                e_F.extend(contr.e_F.tolist())

                # identify friction forces with constant force reservoirs
                for i_N, i_F, force_law in contr.friction_laws:
                    if len(i_N) == 0:
                        self.constant_force_reservoir = True

        self.e_N = np.array(e_N)
        self.e_F = np.array(e_F)

        # call assembler callback: call methods that require first an assembly of the system
        self.assembler_callback()

        # compute consisten initial conditions
        self.q0 = np.array(q0)
        self.u0 = np.array(u0)

        # compute constant system parts
        # - parts of the mass matrix
        coo = CooMatrix((self.nu, self.nu))
        if self.__M_contr:
            I_constant_mass_matrix = np.array(
                [
                    (
                        contr.constant_mass_matrix
                        if hasattr(contr, "constant_mass_matrix")
                        else False
                    )
                    for contr in self.__M_contr
                ]
            )
            self.I_M = ~I_constant_mass_matrix
            self.__M_contr = np.array(self.__M_contr)
            for contr in self.__M_contr[I_constant_mass_matrix]:
                coo[contr.uDOF, contr.uDOF] = contr.M(self.t0, self.q0[contr.qDOF])
        self._M0 = coo.tocoo()

        # - compliance matrix
        coo = CooMatrix((self.nla_c, self.nla_c))
        for contr in self.__c_contr:
            coo[contr.la_cDOF, contr.la_cDOF] = contr.c_la_c()
        self._c_la_c0 = coo.tocoo()

        # compute consistent initial conditions
        (
            self.t0,
            self.q0,
            self.u0,
            self.q_dot0,
            self.u_dot0,
            self.la_g0,
            self.la_gamma0,
            self.la_c0,
            self.la_N0,
            self.la_F0,
        ) = consistent_initial_conditions(self, *args, **kwargs)

    def assembler_callback(self):
        for contr in self.__assembler_callback_contr:
            contr.assembler_callback()

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        q_dot = np.zeros(self.nq, dtype=np.common_type(q, u))
        for contr in self.__q_dot_contr:
            q_dot[contr.my_qDOF] = contr.q_dot(t, q[contr.qDOF], u[contr.uDOF])
        return q_dot

    def q_dot_q(self, t, q, u, format="coo"):
        coo = CooMatrix((self.nq, self.nq))
        for contr in self.__q_dot_q_contr:
            coo[contr.my_qDOF, contr.qDOF] = contr.q_dot_q(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return coo.asformat(format)

    def q_dot_u(self, t, q, format="coo"):
        coo = CooMatrix((self.nq, self.nu))
        for contr in self.__q_dot_u_contr:
            coo[contr.my_qDOF, contr.uDOF] = contr.q_dot_u(t, q[contr.qDOF])
        return coo.asformat(format)

    def step_callback(self, t, q, u):
        for contr in self.__step_callback_contr:
            q[contr.qDOF], u[contr.uDOF] = contr.step_callback(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return q, u

    ################
    # total energies
    ################
    def E_pot(self, t, q):
        E_pot = 0
        for contr in self.__E_pot_contr:
            E_pot += contr.E_pot(t, q[contr.qDOF])
        return E_pot

    def E_kin(self, t, q, u):
        E_kin = 0
        for contr in self.__E_kin_contr:
            E_kin += contr.E_kin(t, q[contr.qDOF], u[contr.uDOF])
        return E_kin

    #####################
    # equations of motion
    #####################
    def M(self, t, q, format="coo"):
        if np.any(self.I_M):
            coo = CooMatrix((self.nu, self.nu))
            for contr in self.__M_contr[self.I_M]:  # only loop over variable mass parts
                coo[contr.uDOF, contr.uDOF] = contr.M(t, q[contr.qDOF])
            return coo.asformat(format) + self._M0.asformat(format)
        else:
            return self._M0.asformat(format)

    def Mu_q(self, t, q, u, format="coo"):
        coo = CooMatrix((self.nu, self.nq))
        for contr in self.__Mu_q_contr:
            coo[contr.uDOF, contr.qDOF] = contr.Mu_q(t, q[contr.qDOF], u[contr.uDOF])
        return coo.asformat(format)

    def h(self, t, q, u):
        h = np.zeros(self.nu, dtype=np.common_type(q, u))
        for contr in self.__h_contr:
            h[contr.uDOF] += contr.h(t, q[contr.qDOF], u[contr.uDOF])
        return h

    def h_q(self, t, q, u, format="coo"):
        coo = CooMatrix((self.nu, self.nq))
        for contr in self.__h_q_contr:
            coo[contr.uDOF, contr.qDOF] = contr.h_q(t, q[contr.qDOF], u[contr.uDOF])
        return coo.asformat(format)

    def h_u(self, t, q, u, format="coo"):
        coo = CooMatrix((self.nu, self.nu))
        for contr in self.__h_u_contr:
            coo[contr.uDOF, contr.uDOF] = contr.h_u(t, q[contr.qDOF], u[contr.uDOF])
        return coo.asformat(format)

    ############
    # compliance
    ############
    def la_c(self, t, q, u):
        la_c = np.zeros(self.nla_c, dtype=np.common_type(q, u))
        for contr in self.__c_contr:
            la_c[contr.la_cDOF] = contr.la_c(t, q[contr.qDOF], u[contr.uDOF])
        return la_c

    def c(self, t, q, u, la_c):
        c = np.zeros(self.nla_c, dtype=np.common_type(q, u, la_c))
        for contr in self.__c_contr:
            c[contr.la_cDOF] = contr.c(
                t, q[contr.qDOF], u[contr.uDOF], la_c[contr.la_cDOF]
            )
        return c

    def c_q(self, t, q, u, la_c, format="coo"):
        coo = CooMatrix((self.nla_c, self.nq))
        for contr in self.__c_q_contr:
            coo[contr.la_cDOF, contr.qDOF] = contr.c_q(
                t, q[contr.qDOF], u[contr.uDOF], la_c[contr.la_cDOF]
            )
        return coo.asformat(format)

    def c_u(self, t, q, u, la_c, format="coo"):
        coo = CooMatrix((self.nla_c, self.nu))
        for contr in self.__c_u_contr:
            coo[contr.la_cDOF, contr.uDOF] = contr.c_u(
                t, q[contr.qDOF], u[contr.uDOF], la_c[contr.la_cDOF]
            )
        return coo.asformat(format)

    def c_la_c(self, format="coo"):
        return self._c_la_c0.asformat(format)

    def W_c(self, t, q, format="coo"):
        coo = CooMatrix((self.nu, self.nla_c))
        for contr in self.__c_contr:
            coo[contr.uDOF, contr.la_cDOF] = contr.W_c(t, q[contr.qDOF])
        return coo.asformat(format)

    def Wla_c_q(self, t, q, la_c, format="coo"):
        coo = CooMatrix((self.nu, self.nq))
        for contr in self.__c_q_contr:
            coo[contr.uDOF, contr.qDOF] = contr.Wla_c_q(
                t, q[contr.qDOF], la_c[contr.la_cDOF]
            )
        return coo.asformat(format)

    ###########
    # actuators
    ###########
    def W_tau(self, t, q, format="coo"):
        coo = CooMatrix((self.nu, self.nla_tau))
        for contr in self.__la_tau_contr:
            coo[contr.uDOF, contr.la_tauDOF] = contr.W_tau(t, q[contr.qDOF])
        return coo.asformat(format)

    def la_tau(self, t, q, u):
        la_tau = np.zeros(self.nla_tau, dtype=np.common_type(q, u))
        for contr in self.__la_tau_contr:
            la_tau[contr.la_tauDOF] = contr.la_tau(t, q[contr.qDOF], u[contr.uDOF])
        return la_tau

    def Wla_tau_q(self, t, q, u, format="coo"):
        coo = CooMatrix((self.nu, self.nq))
        for contr in self.__la_tau_contr:
            coo[contr.uDOF, contr.qDOF] = contr.Wla_tau_q(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return coo.asformat(format)

    def Wla_tau_u(self, t, q, u, format="coo"):
        coo = CooMatrix((self.nu, self.nu))
        for contr in self.__la_tau_contr:
            coo[contr.uDOF, contr.uDOF] = contr.Wla_tau_u(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return coo.asformat(format)

    def tau(self, t):
        tau = np.zeros(self.ntau)
        for contr in self.__tau_contr:
            tau[contr.tauDOF] = contr.tau(t)
        return tau

    def set_tau(self, tau):
        if callable(tau):
            for contr in self.__tau_contr:
                contr.tau = lambda t: tau(t)[contr.tauDOF]
        else:
            for contr in self.__tau_contr:
                contr.tau = lambda t: tau[contr.tauDOF]

    def set_tau_from_dict(self, tau_dict):
        raise NotImplementedError
        # this is not tested!
        for name, tau in tau_dict.items():
            contr = self.contributions_map[name]
            if callable(tau):
                contr.tau = lambda t: tau(t)[contr.tauDOF]
            else:
                contr.tau = lambda t: tau[contr.tauDOF]

    #########################################
    # bilateral constraints on position level
    #########################################
    def g(self, t, q):
        g = np.zeros(self.nla_g, dtype=q.dtype)
        for contr in self.__g_contr:
            g[contr.la_gDOF] = contr.g(t, q[contr.qDOF])
        return g

    def g_q(self, t, q, format="coo"):
        coo = CooMatrix((self.nla_g, self.nq))
        for contr in self.__g_contr:
            coo[contr.la_gDOF, contr.qDOF] = contr.g_q(t, q[contr.qDOF])
        return coo.asformat(format)

    def g_q_T_mu_q(self, t, q, mu_g, format="coo"):
        coo = CooMatrix((self.nq, self.nq))
        for contr in self.__g_contr:
            coo[contr.qDOF, contr.qDOF] = contr.g_q_T_mu_q(
                t, q[contr.qDOF], mu_g[contr.la_gDOF]
            )
        return coo.asformat(format)

    def W_g(self, t, q, format="coo"):
        coo = CooMatrix((self.nu, self.nla_g))
        for contr in self.__g_contr:
            coo[contr.uDOF, contr.la_gDOF] = contr.W_g(t, q[contr.qDOF])
        return coo.asformat(format)

    def Wla_g_q(self, t, q, la_g, format="coo"):
        coo = CooMatrix((self.nu, self.nq))
        for contr in self.__g_contr:
            coo[contr.uDOF, contr.qDOF] = contr.Wla_g_q(
                t, q[contr.qDOF], la_g[contr.la_gDOF]
            )
        return coo.asformat(format)

    def g_dot(self, t, q, u):
        g_dot = np.zeros(self.nla_g, dtype=np.common_type(q, u))
        for contr in self.__g_contr:
            g_dot[contr.la_gDOF] = contr.g_dot(t, q[contr.qDOF], u[contr.uDOF])
        return g_dot

    # TODO: Assemble chi_g for efficiency
    def chi_g(self, t, q):
        return self.g_dot(t, q, np.zeros(self.nu))

    def g_dot_u(self, t, q, format="coo"):
        coo = CooMatrix((self.nla_g, self.nu))
        for contr in self.__g_contr:
            coo[contr.la_gDOF, contr.uDOF] = contr.g_dot_u(t, q[contr.qDOF])
        return coo.asformat(format)

    def g_dot_q(self, t, q, u, format="coo"):
        coo = CooMatrix((self.nla_g, self.nq))
        for contr in self.__g_contr:
            coo[contr.la_gDOF, contr.qDOF] = contr.g_dot_q(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return coo.asformat(format)

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g, dtype=np.common_type(q, u, u_dot))
        for contr in self.__g_contr:
            g_ddot[contr.la_gDOF] = contr.g_ddot(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
        return g_ddot

    # TODO: Assemble zeta_g for efficency
    def zeta_g(self, t, q, u):
        return self.g_ddot(t, q, u, np.zeros(self.nu))

    #########################################
    # bilateral constraints on velocity level
    #########################################
    def gamma(self, t, q, u):
        gamma = np.zeros(self.nla_gamma, dtype=np.common_type(q, u))
        for contr in self.__gamma_contr:
            gamma[contr.la_gammaDOF] = contr.gamma(t, q[contr.qDOF], u[contr.uDOF])
        return gamma

    # TODO: Assemble chi_gamma for efficency
    def chi_gamma(self, t, q):
        return self.gamma(t, q, np.zeros(self.nu))

    def gamma_q(self, t, q, u, format="coo"):
        coo = CooMatrix((self.nla_gamma, self.nq))
        for contr in self.__gamma_contr:
            coo[contr.la_gammaDOF, contr.qDOF] = contr.gamma_q(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return coo.asformat(format)

    def gamma_u(self, t, q, format="coo"):
        coo = CooMatrix((self.nla_gamma, self.nu))
        for contr in self.__gamma_contr:
            coo[contr.la_gammaDOF, contr.uDOF] = contr.gamma_u(t, q[contr.qDOF])
        return coo.asformat(format)

    def gamma_dot(self, t, q, u, u_dot):
        gamma_dot = np.zeros(self.nla_gamma, dtype=np.common_type(q, u, u_dot))
        for contr in self.__gamma_contr:
            gamma_dot[contr.la_gammaDOF] = contr.gamma_dot(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
        return gamma_dot

    def gamma_dot_q(self, t, q, u, u_dot, format="coo"):
        coo = CooMatrix((self.nla_gamma, self.nq))
        for contr in self.__gamma_contr:
            coo[contr.la_gammaDOF, contr.qDOF] = contr.gamma_dot_q(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
        return coo.asformat(format)

    def gamma_dot_u(self, t, q, u, u_dot, format="coo"):
        coo = CooMatrix((self.nla_gamma, self.nu))
        for contr in self.__gamma_contr:
            coo[contr.la_gammaDOF, contr.uDOF] = contr.gamma_dot_u(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
        return coo.asformat(format)

    # TODO: Assemble zeta_gamma for efficency
    def zeta_gamma(self, t, q, u):
        return self.gamma_dot(t, q, u, np.zeros(self.nu))

    def W_gamma(self, t, q, format="coo"):
        coo = CooMatrix((self.nu, self.nla_gamma))
        for contr in self.__gamma_contr:
            coo[contr.uDOF, contr.la_gammaDOF] = contr.W_gamma(t, q[contr.qDOF])
        return coo.asformat(format)

    def Wla_gamma_q(self, t, q, la_gamma, format="coo"):
        coo = CooMatrix((self.nu, self.nq))
        for contr in self.__gamma_contr:
            coo[contr.uDOF, contr.qDOF] = contr.Wla_gamma_q(
                t, q[contr.qDOF], la_gamma[contr.la_gammaDOF]
            )
        return coo.asformat(format)

    #####################################################
    # stabilization conditions for the kinematic equation
    #####################################################
    def g_S(self, t, q):
        g_S = np.zeros(self.nla_S, dtype=q.dtype)
        for contr in self.__g_S_contr:
            g_S[contr.la_SDOF] = contr.g_S(t, q[contr.qDOF])
        return g_S

    def g_S_q(self, t, q, format="coo"):
        coo = CooMatrix((self.nla_S, self.nq))
        for contr in self.__g_S_contr:
            coo[contr.la_SDOF, contr.qDOF] = contr.g_S_q(t, q[contr.qDOF])
        return coo.asformat(format)

    #################
    # normal contacts
    #################
    def g_N(self, t, q):
        g_N = np.zeros(self.nla_N, dtype=q.dtype)
        for contr in self.__g_N_contr:
            g_N[contr.la_NDOF] = contr.g_N(t, q[contr.qDOF])
        return g_N

    def g_N_q(self, t, q, format="coo"):
        coo = CooMatrix((self.nla_N, self.nq))
        for contr in self.__g_N_contr:
            coo[contr.la_NDOF, contr.qDOF] = contr.g_N_q(t, q[contr.qDOF])
        return coo.asformat(format)

    def W_N(self, t, q, format="coo"):
        coo = CooMatrix((self.nu, self.nla_N))
        for contr in self.__g_N_contr:
            coo[contr.uDOF, contr.la_NDOF] = contr.W_N(t, q[contr.qDOF])
        return coo.asformat(format)

    def g_N_dot(self, t, q, u):
        g_N_dot = np.zeros(self.nla_N, dtype=np.common_type(q, u))
        for contr in self.__g_N_contr:
            g_N_dot[contr.la_NDOF] = contr.g_N_dot(t, q[contr.qDOF], u[contr.uDOF])
        return g_N_dot

    def g_N_ddot(self, t, q, u, u_dot):
        g_N_ddot = np.zeros(self.nla_N, dtype=np.common_type(q, u, u_dot))
        for contr in self.__g_N_contr:
            g_N_ddot[contr.la_NDOF] = contr.g_N_ddot(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
        return g_N_ddot

    def xi_N(self, t_pre, t_post, q_pre, q_post, u_pre, u_post):
        xi_N = np.zeros(self.nla_N, dtype=np.common_type(q_post, u_post))
        for contr in self.__g_N_contr:
            xi_N[contr.la_NDOF] = contr.g_N_dot(
                t_post, q_post[contr.qDOF], u_post[contr.uDOF]
            ) + contr.e_N * contr.g_N_dot(t_pre, q_pre[contr.qDOF], u_pre[contr.uDOF])
        return xi_N

    def xi_N_q(self, t_post, q_post, u_post, format="coo"):
        coo = CooMatrix((self.nla_N, self.nq))
        for contr in self.__g_N_contr:
            coo[contr.la_NDOF, contr.qDOF] = contr.g_N_dot_q(
                t_post, q_post[contr.qDOF], u_post[contr.uDOF]
            )
        return coo.asformat(format)

    # TODO: Assemble chi_N for efficency
    def chi_N(self, t, q):
        return self.g_N_dot(t, q, np.zeros(self.nu), dtype=q.dtype)

    def g_N_dot_u(self, t, q, format="coo"):
        warnings.warn(
            "We assume g_N_dot_u(t, q) == W_N(t, q).T. This function will be deleted soon!"
        )
        coo = CooMatrix((self.nla_N, self.nu))
        for contr in self.__g_N_contr:
            coo[contr.la_NDOF, contr.uDOF] = contr.g_N_dot_u(t, q[contr.qDOF])
        return coo.asformat(format)

    def Wla_N_q(self, t, q, la_N, format="coo"):
        coo = CooMatrix((self.nu, self.nq))
        for contr in self.__g_N_contr:
            coo[contr.uDOF, contr.qDOF] = contr.Wla_N_q(
                t, q[contr.qDOF], la_N[contr.la_NDOF]
            )
        return coo.asformat(format)

    #################
    # friction
    #################
    def gamma_F(self, t, q, u):
        gamma_F = np.zeros(self.nla_F, dtype=np.common_type(q, u))
        for contr in self.__gamma_F_contr:
            gamma_F[contr.la_FDOF] = contr.gamma_F(t, q[contr.qDOF], u[contr.uDOF])
        return gamma_F

    def gamma_F_dot(self, t, q, u, u_dot):
        gamma_F_dot = np.zeros(self.nla_F, dtype=np.common_type(q, u, u_dot))
        for contr in self.__gamma_F_contr:
            gamma_F_dot[contr.la_FDOF] = contr.gamma_F_dot(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
        return gamma_F_dot

    def xi_F(self, t_pre, t_post, q_pre, q_post, u_pre, u_post):
        xi_F = np.zeros(self.nla_F, dtype=np.common_type(q_post, u_post))
        for contr in self.__gamma_F_contr:
            xi_F[contr.la_FDOF] = contr.gamma_F(
                t_post, q_post[contr.qDOF], u_post[contr.uDOF]
            ) + contr.e_F * contr.gamma_F(t_pre, q_pre[contr.qDOF], u_pre[contr.uDOF])
        return xi_F

    def xi_F_q(self, t_post, q_post, u_post, format="coo"):
        coo = CooMatrix((self.nla_F, self.nq))
        for contr in self.__gamma_F_contr:
            coo[contr.la_FDOF, contr.qDOF] = contr.gamma_F_q(
                t_post, q_post[contr.qDOF], u_post[contr.uDOF]
            )
        return coo.asformat(format)

    def gamma_F_q(self, t, q, u, format="coo"):
        coo = CooMatrix((self.nla_F, self.nq))
        for contr in self.__gamma_F_q_contr:
            coo[contr.la_FDOF, contr.qDOF] = contr.gamma_F_q(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return coo.asformat(format)

    def gamma_F_u(self, t, q, format="coo"):
        warnings.warn(
            "We assume gamma_F_u(t, q) == W_F(t, q).T. This function will be deleted soon!"
        )
        coo = CooMatrix((self.nla_F, self.nu))
        for contr in self.__gamma_F_contr:
            coo[contr.la_FDOF, contr.uDOF] = contr.gamma_F_u(t, q[contr.qDOF])
        return coo.asformat(format)

    def gamma_F_dot_q(self, t, q, u, u_dot, format="coo"):
        coo = CooMatrix((self.nla_F, self.nq))
        for contr in self.__gamma_F_contr:
            coo[contr.la_FDOF, contr.qDOF] = contr.gamma_F_dot_q(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
        return coo.asformat(format)

    def gamma_F_dot_u(self, t, q, u, u_dot, format="coo"):
        coo = CooMatrix((self.nla_F, self.nu))
        for contr in self.__gamma_F_contr:
            coo[contr.la_FDOF, contr.uDOF] = contr.gamma_F_dot_u(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
        return coo.asformat(format)

    def W_F(self, t, q, format="coo"):
        coo = CooMatrix((self.nu, self.nla_F))
        for contr in self.__gamma_F_contr:
            coo[contr.uDOF, contr.la_FDOF] = contr.W_F(t, q[contr.qDOF])
        return coo.asformat(format)

    def Wla_F_q(self, t, q, la_F, format="coo"):
        coo = CooMatrix((self.nu, self.nq))
        for contr in self.__gamma_F_contr:
            coo[contr.uDOF, contr.qDOF] = contr.Wla_F_q(
                t, q[contr.qDOF], la_F[contr.la_FDOF]
            )
        return coo.asformat(format)

    #########################
    # general linearization #
    #########################
    def D_bar(self, t, q, u):
        return -self.h_u(t, q, u)

    def K_bar(self, t, q, u, u_dot, la_g, la_gamma, la_c):
        part_mass = self.Mu_q(t, q, u_dot)
        part_h = -self.h_q(t, q, u)
        part_g = -self.Wla_g_q(t, q, la_g)
        part_gamma = -self.Wla_gamma_q(t, q, la_gamma)
        part_c = -self.Wla_c_q(t, q, la_c)
        return part_mass + part_h + part_g + part_gamma + part_c

    #################################################
    # projection of constraint of non-holonimc type #
    #################################################
    def G(self, t, q, u, format="coo"):
        coo = CooMatrix((self.nu, self.nu))
        for contr in self.__q_dot_u_contr:
            Gi = contr.G(t, q[contr.qDOF], u[contr.uDOF])
            coo[contr.uDOF, contr.uDOF] = Gi
        return coo.asformat(format)

    def G_dot(self, t, q, u, u_dot, format="coo"):
        coo = CooMatrix((self.nu, self.nu))
        for contr in self.__q_dot_u_contr:
            Gi_dot = contr.G_dot(t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF])
            coo[contr.uDOF, contr.uDOF] = Gi_dot
        return coo.asformat(format)

    def linearize(
        self, t, q, u, static_eq=False, constraints=None, debug=False, ccic=True
    ):
        # ccic: compute_consistent_initial_conditions
        if ccic:
            # save state
            t0_ = self.t0
            q0_ = self.q0
            u0_ = self.u0

            # set stuff for consistent initial condition
            self.t0 = t
            self.q0 = q
            self.u0 = u

            # get consistent initial conditions
            t, q, u, q_dot, u_dot, la_g, la_gamma, la_c, la_N, la_F = (
                consistent_initial_conditions(self)
            )

            # use old initial state again
            self.t0 = t0_
            self.q0 = q0_
            self.u0 = u0_

        else:
            # TODO: may get them from kwargs
            u_dot = np.zeros([self.nu], dtype=float)
            la_g = np.zeros([self.nla_g], dtype=float)
            la_gamma = np.zeros([self.nla_gamma], dtype=float)
            la_c = np.zeros([self.nla_c], dtype=float)

        # continue with linearization
        M0 = self.M(t, q)
        D0_bar = self.D_bar(t, q, u)
        K0_bar = self.K_bar(t, q, u, u_dot, la_g, la_gamma, la_c)

        # project out the contraints g_S
        if static_eq:
            G0 = CooMatrix((self.nu, self.nu)).asformat("coo")
            G0_dot = CooMatrix((self.nu, self.nu)).asformat("coo")
            assert constraints in [None, "NullSpace", "ComplianceProjection"]
        else:
            G0 = self.G(t, q, u)
            G0_dot = self.G_dot(t, q, u, u_dot)
            # TODO: I still don't know how to handle constraints for a general solution
            assert constraints in [None, "NullSpace", "ComplianceProjection"]
        B0 = self.q_dot_u(t, q)

        # handle compliance form
        # TODO: it might be benefitial to implement the inverse of c_la_c directly in the contributions
        c_la_c = self.c_la_c("csc")
        if c_la_c.shape[0] > 0:
            K0_c = (
                self.W_c(t, q)
                @ scipy.sparse.linalg.inv(c_la_c)
                @ self.c_q(t, q, u, la_c)
            )
        else:
            K0_c = CooMatrix((self.nu, self.nq)).asformat("csr")

        D0 = D0_bar + M0 @ G0
        K0 = (K0_bar + K0_c) @ B0 + D0_bar @ G0 + M0 @ G0_dot

        # # from a M-D-G-K-N system
        # M = M0
        # D = 0.5 * (D0 + D0.T)
        # G = 0.5 * (D0 - D0.T)
        # K = 0.5 * (K0 + K0.T)
        # N = 0.5 * (K0 - K0.T)

        if (constraints is None) or (self.nla_g + self.nla_gamma == 0):
            return (M0, D0, K0), B0, None

        W_g = self.W_g(t, q)
        W_gamma = self.W_gamma(t, q)
        W = scipy.sparse.hstack([W_g, W_gamma])
        g_q = self.g_q(t, q)
        if constraints == "NullSpace":
            # project out the contraints g
            # as W_g is sparse, maybe something from here could be helpfull to get a sparse matrix
            # https://stackoverflow.com/questions/33410146/how-can-i-compute-the-null-space-kernel-x-m-x-0-of-a-sparse-matrix-in-pytho

            Bg_proj_dense = scipy.linalg.null_space(W.T.toarray())
            Bg_proj = scipy.sparse.csr_matrix(Bg_proj_dense)

            las_p = None

            if debug:
                print(f"W_g.T @ Nullspace: {scipy.sparse.linalg.norm(W.T @ Bg_proj)}")

        elif constraints == "ComplianceProjection":
            W_gamma = self.W_gamma(t, q)
            # TODO: what K0_g matrix to use?
            # TODO: do we even have to use M0?
            K0_g = W_g @ W_g.T + W_gamma @ W_gamma.T

            # TODO: maybe choose different 'stiffnesses' for g and gamma
            K_compliant = K0_g * 1e9

            M_arr = M0.toarray()
            K_arr = K_compliant.toarray()
            las_g, Vs_g = [
                np.real_if_close(i) for i in scipy.linalg.eigh(-K_arr, M_arr)
            ]

            sort_idx = np.argsort(-las_g)
            las_g = las_g[sort_idx]
            Vs_g = Vs_g[:, sort_idx]

            # modify projecting matrices
            M_proj = np.eye(self.nu, dtype=float)
            # M_proj = M0.toarray()
            K_proj = np.diag(np.arange(self.nu, dtype=float) + 1)
            # TODO: We can use a "negative stiffness" in K_proj. So that we can filter positive and negative eigenvalues
            # K_proj = np.zeros((self.nu, self.nu))
            l_p, V_p = [
                np.real_if_close(i) for i in scipy.linalg.eigh(-K_proj - K_arr, M_proj)
            ]

            sort_idx = np.argsort(-l_p)
            l_p = l_p[sort_idx]
            V_p = V_p[:, sort_idx]

            if debug:
                np.set_printoptions(linewidth=300)
                print(l_p)
                print(V_p[:, 0])  # / V_p[0, 0])

            las_g = l_p
            Vs_g = V_p

            n_min = self.nu - self.nla_g - self.nla_gamma
            las_p = l_p[:n_min]
            # debug prints
            if False:
                for i in range(n_min):
                    if not np.isclose(las_g[i], 0.0):
                        print(f"{i = }, la: {las_g[i]:.3e} (should be close to 0)")

                    off = np.linalg.norm(W_g.T @ Vs_g[:, i])
                    if not np.isclose(off, 0.0):
                        print(f"{i = }, g_violation: {off:.3e} (should be close to 0)")

                for i in range(n_min, self.nu):
                    if np.isclose(las_g[i], 0.0):
                        print(f"{i = }, la: {las_g[i]:.3e} (should not be close to 0)")

                    off = np.linalg.norm(W_g.T @ Vs_g[:, i])
                    if np.isclose(off, 0.0):
                        print(
                            f"{i = }, g_violation: {off:.3e} (should not be close to 0)"
                        )

            if debug:
                # Nullspace violation of W
                Nv_ad = W.T @ Vs_g[:, n_min - 1]
                Nv_nad = W.T @ Vs_g[:, n_min]
                r_la = 0 if n_min == 0 else las_g[n_min - 1] / las_g[n_min]
                r_Nv = (
                    0 if n_min == 0 else np.linalg.norm(Nv_ad) / np.linalg.norm(Nv_nad)
                )
                print(f"Projection ratio la: {r_la:.3e}")
                print(f"Projection ratio NullSpace_violation (W): {r_Nv:.3e}")

            Bg_proj = scipy.sparse.csr_matrix(Vs_g[:, :n_min])

        # compute reduced matrices
        Mg = Bg_proj.T @ M0 @ Bg_proj
        Dg = Bg_proj.T @ D0 @ Bg_proj
        Kg = Bg_proj.T @ K0 @ Bg_proj
        Bg = B0 @ Bg_proj

        # check if the reduced directions fulfill the constraints
        for i, q_dir in enumerate(Bg.T):
            off = np.linalg.norm(g_q @ q_dir.toarray())
            if not np.isclose(off, 0.0):
                print(f"{i = }, {off}")

        return (Mg, Dg, Kg), Bg, las_p

    def new_eigenmodes(self, solution, idx):
        t = solution.t[idx]
        q = solution.q[idx]
        la_c = solution.la_c[idx]
        la_g = solution.la_g[idx]

        u = np.zeros([self.nu], dtype=float)
        u_dot = np.zeros([self.nu], dtype=float)
        la_gamma = np.zeros([self.nla_gamma], dtype=float)

        # naming convention of generalized vectors:
        # nonlinear vectors:
        #   q
        #   u
        #
        # after projection of g_S:
        #   dq = B0 @ dz
        #   du = dz_dot
        #
        # after projection of all bilateral constraints
        #   dq = B0 @ T_hat @ dz_hat
        #   dz = T_hat @ dz_hat
        #        T_hat = T_bil @ T_int

        # naming convention of matrices:
        # 1st step, straight forward linearization:
        #   M0 @ du_dot + D0_bar @ du + K0_bar @ dq = Wc0 @ dla_c + Wg0 @ dla_g
        #   dq_dot = B0 @ du + H0 @ dq
        #   Kc^-1 @ dla_c + lc_q @ dq = 0
        #
        # 2nd step, projection of g_S and solving of compliance:
        #   M0 @ dz_ddot + D0 @ dz_dot + K0 @ dz = Wg0 @ dla_g
        #   D0 = D0_bar + ... dynamic
        #   K0 = (K0_bar + Wc0 @ Kc @ lc_q) @ B0 + ... dynamic
        #
        # 3rd step, projection of g
        #   M_hat @ dz_hat_ddot + D_hat @ dz_hat_dot + K_hat @ dz_hat = 0
        #       M_hat = T_hat.T @ M0 @ T_hat
        #       D_hat = T_hat.T @ D0 @ T_hat
        #       K_hat = T_hat.T @ K0 @ T_hat
        #   B_hat = B0 @ T_hat, T_hat = T_int @ T_bil

        ###############################################
        # 1st: compute straight forward linearization #
        ###############################################
        # and check on the run for static equilibrium
        # force equation
        M0 = self.M(t, q)
        # D0_bar =
        K0_bar = self.K_bar(t, q, u, u_dot, la_g, la_gamma, la_c)
        h0 = self.h(t, q, u)

        # kinematics equation
        B0 = self.q_dot_u(t, q, format="csc")
        # H0 =

        # compliance
        # TODO: it might be benefitial to implement the inverse of c_la_c directly in the contributions
        Wc0 = self.W_c(t, q, format="csc")
        Kc_inv = self.c_la_c("csc")
        c0 = self.c(t, q, u, la_c)
        c_q0 = self.c_q(t, q, u, la_c)
        if self.nla_c > 0:
            K0_c_bar = Wc0 @ scipy.sparse.linalg.inv(Kc_inv) @ c_q0
        else:
            K0_c_bar = CooMatrix((self.nu, self.nq)).asformat("csr")

        # bilateral constraints, position level
        Wg0 = self.W_g(t, q, format="csr")
        g0 = self.g(t, q)
        g_q0 = self.g_q(t, q, format="csc")

        # bilateral constraints, velocity level
        # Wgamma0 =
        # gamma0 =

        # check for static equilibrium
        f0 = h0 + Wc0 @ la_c + Wg0 @ la_g
        assert np.isclose(np.linalg.norm(f0), 0.0), "No static equilibrium point!"
        assert np.isclose(np.linalg.norm(c0), 0.0), "No static equilibrium point!"
        assert np.isclose(np.linalg.norm(g0), 0.0), "No static equilibrium point!"
        # gamma0

        #############################################
        # 2nd: project out g_S and solve compliance #
        #############################################
        # D0 =
        K0 = (K0_bar + K0_c_bar) @ B0

        ##########################################
        # 3rd: project out bilateral constraints #
        ##########################################
        # get list of all contributions with bilateral constraints
        # TODO: add constraints on velocity level

        # check if contribution has own DOFs for q and u
        internal_contr, non_internal_contr = [], []
        for contr in self.__g_contr:
            # TODO: maybe just check if there is a "T" attribute
            if hasattr(contr, "nq") and hasattr(contr, "nu"):
                internal_contr.append(contr)
            else:
                non_internal_contr.append(contr)

        ########################################
        # A: constraints inside a contribution #
        ########################################
        nla_g_intern = np.sum([c.nla_g for c in internal_contr])
        T_int, col = CooMatrix((self.nu, self.nu - nla_g_intern)), 0
        removed_laDOFs, changing_uDOFs = [], []
        for contr in internal_contr:
            if hasattr(contr, "T"):
                # if there is an implementation
                T = contr.T(t, q, format="csc")
            else:
                # project numerically using W_g
                W_g_contrT = contr.W_g(t, q[contr.qDOF]).toarray().T
                T = scipy.sparse.csc_array(scipy.linalg.null_space(W_g_contrT))

            ni_contr = contr.nu - contr.nla_g
            T_int[contr.uDOF, col : col + ni_contr] = T
            col += ni_contr

            removed_laDOFs.extend(contr.la_gDOF)
            changing_uDOFs.extend(contr.uDOF)

        # double check if no wrong DOF was touched
        removed_laDOFs = np.array(removed_laDOFs)
        changing_uDOFs = np.array(changing_uDOFs)
        assert len(removed_laDOFs) == len(
            np.unique(removed_laDOFs)
        ), "Some contributions were working on the same laDOF."
        assert len(changing_uDOFs) == len(
            np.unique(changing_uDOFs)
        ), "Some contributions were working on the same uDOF."

        # these are uDOFs by rigid bodies, rods without constraints, ...
        unchanging_uDOFs = np.setdiff1d(np.arange(self.nu), changing_uDOFs)
        T_int[unchanging_uDOFs, unchanging_uDOFs] = scipy.sparse.eye_array(
            len(unchanging_uDOFs), dtype=float
        )
        T_int = T_int.asformat("csc")

        ######################################
        # B: remaining bilateral constraints #
        ######################################
        # try straight forward Nullspace matrix on W_g.T
        non_internal_laDOFs = np.setdiff1d(np.arange(self.nla_g), removed_laDOFs)
        W_g_non_internalT = Wg0[:, non_internal_laDOFs].T.toarray()
        T_bil = scipy.sparse.csc_array(
            scipy.linalg.null_space(W_g_non_internalT @ T_int)
        )

        if True:
            # print error to the original set
            # equation of motion
            zero0 = T_bil.T @ T_int.T @ Wg0

            # constraint equation
            zero1 = g_q0 @ B0 @ T_int @ T_bil

            print(f"Constraint projection errors:")
            print(f"norm(T_bil.T @ T_int.T @ W_g) : {scipy.sparse.linalg.norm(zero0)}")
            print(f"norm(g_q @ B @ T_int @ T_bil) : {scipy.sparse.linalg.norm(zero1)}")
            print()

        if False:
            # remove the other constrained DOFs
            uDOFs = np.setdiff1d(np.arange(self.nu), remove_uDOFs)
            B_red = B0[:, uDOFs]

            # TODO: get this together with remove_uDOFs
            d_uDOFs = dependent_uDOFs
            i_uDOFs = np.setdiff1d(np.arange(self.nu), d_uDOFs)
            assert self.nla_g == len(d_uDOFs)
            ni = len(i_uDOFs)

            # try it with W_g
            Wg0 = self.W_g(t, q, format="csr")
            W_gi = Wg0[i_uDOFs, :]
            W_gd = Wg0[d_uDOFs, :]

            T_W_g = CooMatrix((self.nu, ni))
            T_W_g[i_uDOFs, :] = scipy.sparse.eye_array(ni, dtype=float)
            T_W_g[d_uDOFs, :] = -scipy.sparse.linalg.inv(W_gd.T) @ W_gi.T
            T_W_g = T_W_g.asformat("csc")

            # try it with g_q @ B
            g_q0 = self.g_q(t, q, format="csc")
            Bi = B0[:, i_uDOFs]
            Bd = B0[:, d_uDOFs]
            T_g = CooMatrix((self.nu, ni))
            T_g[i_uDOFs, :] = scipy.sparse.eye_array(ni, dtype=float)
            T_g[d_uDOFs, :] = -scipy.sparse.linalg.inv(g_q0 @ Bd) @ g_q0 @ Bi
            T_g = T_g.asformat("csc")

            # try straight forward Nullspace matrix on W_g.T
            T_N_W_g = scipy.sparse.csc_array(scipy.linalg.null_space(Wg0.toarray().T))

            # try straight forward Nullspace matrix on g_q @ B
            T_N_g = scipy.sparse.csc_array(scipy.linalg.null_space(g_q0 @ B0.toarray()))

            print(f"{self.nq = }, {self.nu = }, {self.nla_g = }")

            # check if T is the nullspace, such that T.T @ W_g = 0 and g_q @ T @ B = 0

            T_matrices = [T_W_g, T_g, T_N_W_g, T_N_g]
            T_names = ["inv W_g", "inv g_q", "N W_g", "N g_q"]
            for T, name in zip(T_matrices, T_names):
                # equation of motion
                zero0 = T.T @ Wg0

                # constraint equation
                zero1 = g_q0 @ B0 @ T

                print(f"\n   {name}")
                print(f"norm(T.T @ W_g)   : {scipy.sparse.linalg.norm(zero0)}")
                print(f"norm(g_q @ B @ T) : {scipy.sparse.linalg.norm(zero1)}")

        # compose total projection matrices
        # du = T @ dz
        T_hat = T_int @ T_bil

        # manipulate matrices
        M_hat = T_hat.T @ M0 @ T_hat
        # D_hat =
        K_hat = T_hat.T @ K0 @ T_hat

        ###########################
        # 4th: compute eigenmodes #
        ###########################
        # compute squared eigenvalues and make them real
        res = list(scipy.linalg.eig(-K_hat.toarray(), M_hat.toarray()))
        for i, v in enumerate(res):
            imag_norm = np.linalg.norm(np.imag(v))
            total_norm = np.linalg.norm(v)
            if total_norm > 0.0:
                ratio = imag_norm / total_norm
                if ratio >= 1e-5:
                    print(
                        f"arg(a+bi) = {ratio:.2e}. This imaginary part will be discarded!"
                    )
            res[i] = np.real(v)

        las_ud_squared, Vs_ud = res

        # sort eigenvalues such that rigid body modes are first
        sort_idx = np.argsort(-las_ud_squared)
        las_ud_squared = las_ud_squared[sort_idx]
        Vs_ud = Vs_ud[:, sort_idx]

        # compute omegas
        omegas = np.zeros([len(las_ud_squared)])
        modes_dq = B0 @ T_hat @ Vs_ud
        for i, lai in enumerate(las_ud_squared):
            if np.isclose(0.0, lai):
                omegas[i] = 0.0
            elif lai > 0:
                msg = f"Warning: An eigenvalue is larger than 0: lambda = {lai:.3e}. This should not happen."
                warnings.warn(msg)
                omegas[i] = np.sqrt(lai)
            else:
                omegas[i] = np.sqrt(-lai)

        # compose solution object with omegas and modes
        sol = Solution(
            self,
            np.array([t]),
            np.array([q]),
            omegas=np.array([omegas]),
            modes_dq=np.array([modes_dq]),
        )

        return omegas, modes_dq, sol

    def eigenmodes(self, t, q, constraints="ComplianceProjection", ccic=True):
        u = np.zeros(self.nu)
        MDK, B, _ = self.linearize(t, q, u, True, constraints, ccic=ccic)

        # get eigenvalues and eigenvectors of the undamped system
        M = MDK[0]
        D = 0.5 * (MDK[1] + MDK[1].T)
        G = 0.5 * (MDK[1] - MDK[1].T)
        K = 0.5 * (MDK[2] + MDK[2].T)
        N = 0.5 * (MDK[2] - MDK[2].T)

        d = M.shape[0]
        atol = (
            np.finfo(float).eps
            * d
            * np.max([1, scipy.sparse.linalg.norm(K) / scipy.sparse.linalg.norm(M)])
        )
        norm_M = scipy.sparse.linalg.norm(M - M.T)
        assert np.isclose(
            norm_M, 0.0, atol=atol * scipy.sparse.linalg.norm(M)
        ), f"Mass matrix is not symmetric! {norm_M}"

        norm_N = scipy.sparse.linalg.norm(N)
        if not np.isclose(norm_N, 0.0, atol=atol):
            warnings.warn(f"There are circular forces that will be neglected! {norm_N}")

        # compute eigenvalues and eigenvectors
        las_ud_squared, Vs_ud = [
            np.real_if_close(i) for i in scipy.linalg.eigh(-K.toarray(), M.toarray())
        ]

        # TODO: check for proportional damped system, i.e.,
        # Hint:
        # turns out, that we can only visualize eigenmodes of undamped systems or proportional damped systems, i.e., if V diagonalizes M_inv@K (V.T @ M_inv @ K @ V is diagonal) V.T @ M_inv@D @ V is also diagonal
        # otherwise there exists no alpha in C, such that alpha * dq is real, i.e., the eigenmode is not in sync.

        # sort eigenvalues such that rigid body modes are first
        sort_idx = np.argsort(-las_ud_squared)
        las_ud_squared = las_ud_squared[sort_idx]
        Vs_ud = Vs_ud[:, sort_idx]

        # compute omegas
        omegas = np.zeros([len(las_ud_squared)])
        modes_dq = B @ Vs_ud
        for i, lai in enumerate(las_ud_squared):
            if np.isclose(0.0, lai, atol=atol):
                omegas[i] = 0.0
            elif lai > 0:
                msg = f"Warning: An eigenvalue is larger than 0: lambda = {lai:.3e}. This should not happen."
                warnings.warn(msg)
                omegas[i] = np.sqrt(lai)
            else:
                omegas[i] = np.sqrt(-lai)

        # compose solution object with omegas and modes
        sol = Solution(
            self,
            np.array([t]),
            np.array([q]),
            omegas=np.array([omegas]),
            modes_dq=np.array([modes_dq]),
        )

        return omegas, modes_dq, sol

    def fundamental_perturbation_matrix(
        self, sol, constraints="ComplianceProjection", debug=False
    ):
        nt = len(sol.t)
        nz = self.nu - self.nla_g - self.nla_gamma

        las_p = np.zeros([nt, nz])
        Mii = np.zeros([nt, nz, nz])
        Minvii = np.zeros([nt, nz, nz])
        Dii = np.zeros([nt, nz, nz])
        Kii = np.zeros([nt, nz, nz])

        Phi = np.eye(2 * nz, dtype=float)
        dt = sol.t[1] - sol.t[0]

        B_last = None
        for i in range(nt):
            ti = sol.t[i]
            qi = sol.q[i]
            ui = sol.u[i]
            MDK, B, l_p = self.linearize(ti, qi, ui, True, constraints)

            las_p[i] = l_p

            nz = B.shape[1]
            p = np.ones(nz, dtype=float)
            if B_last is not None:
                # multiply directions with -1 if necessary
                for j in range(nz):
                    if B[:, [j]].T @ B_last[:, [j]] < 0:
                        B[:, [j]] *= -1.0
                        p[j] = -1.0

            if constraints == "ComplianceProjection":
                B_last = B

            # apply multiplications with -1 to system matrices
            P = scipy.sparse.diags(p)
            M = P.T @ MDK[0] @ P
            D = P.T @ MDK[1] @ P
            K = P.T @ MDK[2] @ P

            Mii[i] = M.toarray()
            Minvii[i] = np.linalg.inv(M.toarray())
            Dii[i] = D.toarray()
            Kii[i] = K.toarray()

            # midpoint rule
            MinvK = np.linalg.solve(M.toarray(), K.toarray())
            MinvD = np.linalg.solve(M.toarray(), D.toarray())
            A = np.block([[np.zeros((nz, nz)), np.eye(nz)], [-MinvK, -MinvD]])
            Phi += dt * A @ Phi * (0.5 if i in [0, nt - 1] else 1)

        import matplotlib.pyplot as plt

        t = sol.t
        fig, ax = plt.subplots()
        fig.suptitle("These values should be distinguishable apart!")
        ax.plot(t, las_p)

        if debug:
            fig, ax_M = plt.subplots(3, 3)
            fig, ax_Mi = plt.subplots(3, 3)
            fig, ax_D = plt.subplots(3, 3)
            fig, ax_K = plt.subplots(3, 3)
            for i in range(nz):
                for j in range(nz):
                    ax_M[i, j].plot(t, Mii[:, i, j])
                    ax_Mi[i, j].plot(t, Minvii[:, i, j])
                    ax_D[i, j].plot(t, Dii[:, i, j])
                    ax_K[i, j].plot(t, Kii[:, i, j])

            plt.show()

        return Phi

    def eigenvalues(self, t, q, u):
        MDK, B, _ = self.linearize(t, q, u, static_eq=False)
        M0 = MDK[0]
        D0 = MDK[1]
        K0 = MDK[2]

        A_hat = CooMatrix((2 * self.nu, 2 * self.nu))
        A_hat[: self.nu, : self.nu] = -K0
        A_hat[self.nu :, self.nu :] = M0

        B_hat = CooMatrix((2 * self.nu, 2 * self.nu))
        B_hat[: self.nu, : self.nu] = D0
        B_hat[: self.nu, self.nu :] = M0
        B_hat[self.nu :, : self.nu] = M0

        # determine eigenvalues
        A_hat = A_hat.toarray()
        B_hat = B_hat.toarray()
        las, Vs = [np.real_if_close(i) for i in scipy.linalg.eig(A_hat, B_hat)]
        return las, Vs
