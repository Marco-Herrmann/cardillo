from abc import ABC, abstractmethod
import numpy as np

from cardillo.utility.parametrize import parametrize


class RodMaterialModel(ABC):
    """Abstract class for rod material models"""

    @abstractmethod
    def prepare(self, xi):
        """Function that returns everything relevant to compute potential, sigma, ... at xi"""
        ...

    @abstractmethod
    def potential(self, epsilon, epsilon0, prepare):
        """
        Parameters
        ----------
        epsilon : np.ndarray (n, 6) or np.ndarray(6,)
            current strains
        epsilon0 : np.ndarray (n, 6) or np.ndarray(6,)
            reference strains
        prepare : struct from self.prepare(xi)
        """
        ...

    @abstractmethod
    def sigma(self, epsilon, epsilon0, prepare): ...

    @abstractmethod
    def sigma_epsilon(self, epsilon, epsilon0, prepare): ...


class RodMaterialModel_compliance(RodMaterialModel):
    """Abstract class for rod material models in compliance form"""

    @abstractmethod
    def C_inv(self, xi): ...


class Simo1986(RodMaterialModel_compliance):
    def __init__(self, Ei, Fi):
        """
        Material model for shear deformable rod with quadratic strain energy
        function found in Simo1986 (2.8), (2.9) and (2.10).

        Parameters
        ----------
        Ei : np.ndarray (3,) or callable(xi) -> np.ndarray (3,)
            E0: dilatational stiffness, i.e., rigidity with resepct to volumetric change.
            E1: shear stiffness in e_y^B-direction.
            E2: shear stiffness in e_z^B-direction.
        Fi : np.ndarray (3,) or callable(xi) -> np.ndarray (3,)
            F0: torsional stiffness
            F1: flexural stiffness around e_y^B-direction.
            F2: flexural stiffness around e_z^B-direction.

        References
        ----------
        Simo1986 : https://doi.org/10.1016/0045-7825(86)90079-4
        """
        self.Ei = parametrize(Ei)
        self.Fi = parametrize(Fi)

    def prepare(self, xi):
        Ei = self.Ei(xi)
        Fi = self.Fi(xi)
        EiFi = np.hstack([Ei, Fi])
        C_Ga = np.apply_along_axis(np.diag, 1, Ei)
        C_Ka = np.apply_along_axis(np.diag, 1, Fi)
        zeros = np.zeros((len(xi), 3, 3))
        C_inv = np.apply_along_axis(np.diag, 1, 1 / EiFi)

        return dict(
            EiFi=EiFi,
            sigma_epsilon=(C_Ga, zeros, zeros, C_Ka),
            C_inv=C_inv,
        )

    def potential(self, epsilon, epsilon0, prepare):
        d_epsilon = epsilon - epsilon0
        sig = prepare["EiFi"] * d_epsilon
        return 0.5 * np.sum(sig * d_epsilon, axis=1)

    def sigma(self, epsilon, epsilon0, prepare):
        d_epsilon = epsilon - epsilon0
        return prepare["EiFi"] * d_epsilon

    def sigma_epsilon(self, epsilon, epsilon0, prepare):
        return prepare["sigma_epsilon"]

    def C_inv(self, prepare):
        return prepare["C_inv"]


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

    def prepare(self, xi):
        return None

    def potential(self, epsilon, epsilon0, prepare):
        nxi = np.atleast_2d(epsilon).shape[0]
        pot = np.empty(nxi)
        for i in range(nxi):
            Gamma = epsilon[i, :3]
            Gamma0 = epsilon0[i, :3]
            Kappa = epsilon[i, 3:]
            Kappa0 = epsilon0[i, 3:]
            pot[i] = self.old.potential(Gamma, Gamma0, Kappa, Kappa0)
        return pot

    def sigma(self, epsilon, epsilon0, prepare):
        nxi = np.atleast_2d(epsilon).shape[0]
        sigma = np.empty((nxi, 6))
        for i in range(nxi):
            Gamma = epsilon[i, :3]
            Gamma0 = epsilon0[i, :3]
            Kappa = epsilon[i, 3:]
            Kappa0 = epsilon0[i, 3:]
            sigma[i, :3] = self.old.B_n(Gamma, Gamma0, Kappa, Kappa0)
            sigma[i, 3:] = self.old.B_m(Gamma, Gamma0, Kappa, Kappa0)
        return sigma

    def sigma_epsilon(self, epsilon, epsilon0, prepare):
        nxi = np.atleast_2d(epsilon).shape[0]
        n_Ga = np.empty((nxi, 3, 3))
        n_Ka = np.empty((nxi, 3, 3))
        m_Ga = np.empty((nxi, 3, 3))
        m_Ka = np.empty((nxi, 3, 3))
        for i in range(nxi):
            Gamma = epsilon[i, :3]
            Gamma0 = epsilon0[i, :3]
            Kappa = epsilon[i, 3:]
            Kappa0 = epsilon0[i, 3:]
            n_Ga[i] = self.old.B_n_B_Gamma(Gamma, Gamma0, Kappa, Kappa0)
            n_Ka[i] = self.old.B_n_B_Kappa(Gamma, Gamma0, Kappa, Kappa0)
            m_Ga[i] = self.old.B_m_B_Gamma(Gamma, Gamma0, Kappa, Kappa0)
            m_Ka[i] = self.old.B_m_B_Kappa(Gamma, Gamma0, Kappa, Kappa0)

        return n_Ga, n_Ka, m_Ga, m_Ka
