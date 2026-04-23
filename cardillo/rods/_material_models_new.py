from abc import ABC, abstractmethod
import numpy as np


class RodMaterialModel(ABC):
    """Abstract class for rod material models"""

    def prepare_quadrature(self, xi):
        """Function that is called to evaluate local stiffnes matrix at quadrature points"""
        ...

    @abstractmethod
    def potential(self, xi, epsilon, epsilon0):
        """
        Parameters
        ----------
        xi : np.ndarray (n,) or scalar
            rod coordinates
        epsilon : np.ndarray (n, 6) or np.ndarray(6, )
            current strains
        epsilon0 : np.ndarray (n, 6) or np.ndarray(6, )
            reference strains
        """
        ...

    @abstractmethod
    def sigma(self, xi, epsilon, epsilon0): ...

    @abstractmethod
    def sigma_epsilon(self, xi, epsilon, epsilon0): ...


class RodMaterialModel_compliance(RodMaterialModel):
    """Abstract class for rod material models in compliance form"""

    @property
    @abstractmethod
    def C_inv(self, xi): ...


class Simo1986(RodMaterialModel_compliance):
    def __init__(self, Ei, Fi):
        """
        Material model for shear deformable rod with quadratic strain energy
        function found in Simo1986 (2.8), (2.9) and (2.10).

        Parameters
        ----------
        Ei : np.ndarray (3,)
            E0: dilatational stiffness, i.e., rigidity with resepct to volumetric change.
            E1: shear stiffness in e_y^K-direction.
            E2: shear stiffness in e_z^K-direction.
        Fi : np.ndarray (3,)
            F0: torsional stiffness
            F1: flexural stiffness around e_y^K-direction.
            F2: flexural stiffness around e_z^K-direction.

        References
        ----------
        Simo1986 : https://doi.org/10.1016/0045-7825(86)90079-4
        """
        self._Ei = Ei if callable(Ei) else lambda xi: np.tile(Ei, (len(xi), 1))
        self._Fi = Fi if callable(Fi) else lambda xi: np.tile(Fi, (len(xi), 1))

    def prepare_quadrature(self, xi):
        self.Ei_qp = self._Ei(xi)
        self.Fi_qp = self._Fi(xi)
        self.EiFi_qp = np.hstack((self.Ei_qp, self.Fi_qp))
        self.C_Ga = np.apply_along_axis(np.diag, 1, self.Ei_qp)
        self.C_Ka = np.apply_along_axis(np.diag, 1, self.Fi_qp)
        self.zeros_qp = np.zeros((len(xi), 3, 3))
        self.C_inv_qp = np.apply_along_axis(np.diag, 1, 1 / self.EiFi_qp)

    def potential(self, xi, epsilon, epsilon0, quadrature=False):
        d_epsilon = epsilon - epsilon0
        if quadrature:
            EiFi = self.EiFi_qp
        else:
            EiFi = np.hstack(self._Ei(xi), self.Fi(xi))
        sig = EiFi * d_epsilon
        return 0.5 * np.sum(sig * d_epsilon, axis=1)

    def sigma(self, xi, epsilon, epsilon0, quadrature=False):
        d_epsilon = epsilon - epsilon0
        if quadrature:
            EiFi = self.EiFi_qp
        else:
            EiFi = np.hstack(self._Ei(xi), self.Fi(xi))
        return EiFi * d_epsilon

    def sigma_epsilon(self, xi, epsilon, epsilon0, quadrature=False):
        if quadrature:
            sig0_eps0 = self.C_Ga
            sig1_eps1 = self.C_Ka
            sig0_eps1 = sig1_eps0 = self.zeros_qp
        else:
            Ei = self._Ei(xi)
            Fi = self._Fi(xi)
            sig0_eps0 = np.apply_along_axis(np.diag, 1, Ei)
            sig1_eps1 = np.apply_along_axis(np.diag, 1, Fi)
            sig0_eps1 = sig1_eps0 = np.zeros_like(sig0_eps0)

        return sig0_eps0, sig0_eps1, sig1_eps0, sig1_eps1

    def C_inv(self, xi, quadrature=False):
        if quadrature:
            C_inv = self.C_inv_qp
        else:
            EiFi = np.hstack(self._Ei(xi), self._Fi(xi))
            C_inv = np.apply_along_axis(np.diag, 1, 1 / EiFi)
        return C_inv


class Harsch2021(RodMaterialModel):
    def __init__(self, Ei, Fi):
        """
        Material model for shear deformable beam with nonlinear axial deformation.

        Parameters
        ----------
        Ei : np.ndarray(3)
            E0: extensional stiffness, i.e., stiffness in r_OP'-direction.
            E1: shear stiffness in e_y^K-direction.
            E2: shear stiffness in e_z^K-direction.
        Fi : np.ndarray(3)
            F0: torsional stiffness
            F1: flexural stiffness around e_y^K-direction.
            F2: flexural stiffness around e_z^K-direction.

        References
        ----------
        Harsch2021: https://doi.org/10.1177/10812865211000790
        """

        # TODO: do this maybe nicer
        from ._material_models import Harsch2021 as Harsch2021_old

        self.old = Harsch2021_old(Ei, Fi)

    def potential(self, xi, epsilon, epsilon0, quadrature=False):
        pot = np.empty(len(xi))
        for i in len(xi):
            Gamma = epsilon[i, :3]
            Gamma0 = epsilon0[i, :3]
            Kappa = epsilon[i, 3:]
            Kappa0 = epsilon0[i, 3:]
            pot[i] = self.old.potential(Gamma, Gamma0, Kappa, Kappa0)
        return pot

    def sigma(self, xi, epsilon, epsilon0, quadrature=False):
        sigma = np.empty((len(xi), 6))
        for i in len(xi):
            Gamma = epsilon[i, :3]
            Gamma0 = epsilon0[i, :3]
            Kappa = epsilon[i, 3:]
            Kappa0 = epsilon0[i, 3:]
            sigma[i, :3] = self.old.B_n(Gamma, Gamma0, Kappa, Kappa0)
            sigma[i, 3:] = self.old.B_m(Gamma, Gamma0, Kappa, Kappa0)
        return sigma

    def sigma_epsilon(self, xi, epsilon, epsilon0, quadrature=False):
        n_Ga = np.empty((len(xi), 3, 3))
        n_Ka = np.empty((len(xi), 3, 3))
        m_Ga = np.empty((len(xi), 3, 3))
        m_Ka = np.empty((len(xi), 3, 3))
        for i in len(xi):
            Gamma = epsilon[i, :3]
            Gamma0 = epsilon0[i, :3]
            Kappa = epsilon[i, 3:]
            Kappa0 = epsilon0[i, 3:]
            n_Ga[i] = self.old.B_n_B_Gamma(Gamma, Gamma0, Kappa, Kappa0)
            n_Ka[i] = self.old.B_n_B_Kappa(Gamma, Gamma0, Kappa, Kappa0)
            m_Ga[i] = self.old.B_m_B_Gamma(Gamma, Gamma0, Kappa, Kappa0)
            m_Ka[i] = self.old.B_m_B_Kappa(Gamma, Gamma0, Kappa, Kappa0)

        return n_Ga, n_Ka, m_Ga, m_Ka
