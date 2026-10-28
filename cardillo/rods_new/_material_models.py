from abc import ABC, abstractmethod
import numpy as np
from warnings import warn

from cardillo.utility.parametrize import parametrize


class RodMaterialModel(ABC):
    """Abstract class for rod material models"""

    idx_damp = np.array([], dtype=int)
    # C(epsilon) and D are independent of the strains, i.e., the rod's e_la_e
    # and d_la_d are constant and only evaluated once by the system
    constant_C = False
    constant_D = False

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
    def sigma(self, epsilon, epsilon0, epsilon_dot, prepare): ...

    @abstractmethod
    def sigma_epsilon(self, epsilon, epsilon0, epsilon_dot, prepare): ...

    def sigma_epsilon_dot(self, epsilon, epsilon0, epsilon_dot, prepare):
        """Derivative of sigma w.r.t. epsilon_dot, default: no viscous stresses"""
        zeros = np.zeros((len(epsilon), 3, 3))
        return (zeros, zeros, zeros, zeros)


class RodMaterialModel_compliance(RodMaterialModel):
    """Abstract class for rod material models in compliance form"""

    @abstractmethod
    def C_inv(self, xi): ...


class LinearViscousDamping:
    """Linear viscous stresses sigma_d = diag(DEi, DFi) @ epsilon_dot, which can
    be added to any rod material model."""

    # D_inv does not depend on the strains
    constant_D = True

    def _init_damping(self, DEi, DFi):
        self.DEi = parametrize(DEi)
        self.DFi = parametrize(DFi)

        # check which indices are damped (assume that is not changing along the rod)
        DEiDFi0 = np.concatenate([self.DEi(0.0), self.DFi(0.0)])
        self.idx_damp = np.array([i for i in range(6) if DEiDFi0[i] != 0.0], dtype=int)

    def _prepare_damping(self, xi):
        DEi = self.DEi(xi)
        DFi = self.DFi(xi)
        DEiDFi = np.hstack([DEi, DFi])
        D_Ga = np.apply_along_axis(np.diag, 1, DEi)
        D_Ka = np.apply_along_axis(np.diag, 1, DFi)
        zeros = np.zeros((len(xi), 3, 3))
        # invert the invertable ones
        D_inv = 1 / np.where(DEiDFi != 0.0, DEiDFi, np.ones_like(DEiDFi))
        D_inv = np.apply_along_axis(np.diag, 1, D_inv)

        return dict(
            DEiDFi=DEiDFi,
            sigma_epsilon_dot=(D_Ga, zeros, zeros, D_Ka),
            D_inv=D_inv,
        )

    def _sigma_damping(self, epsilon_dot, prepare):
        # epsilon_dot = None: only elastic stresses, e.g., in the compliance form
        if epsilon_dot is None:
            return 0.0
        return prepare["DEiDFi"] * epsilon_dot

    def sigma_epsilon_dot(self, epsilon, epsilon0, epsilon_dot, prepare):
        return prepare["sigma_epsilon_dot"]

    def D_inv(self, prepare):
        return prepare["D_inv"]


class Simo1986(LinearViscousDamping, RodMaterialModel_compliance):
    constant_C = True

    def __init__(self, Ei, Fi, DEi=np.zeros(3), DFi=np.zeros(3)):
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
        DEi : np.ndarray (3,) or callable(xi) -> np.ndarray (3,)
            Damping for dilataion and shear
        DFi : np.ndarray (3,) or callable(xi) -> np.ndarray (3,)
            Damping for torsion and bending
        References
        ----------
        Simo1986 : https://doi.org/10.1016/0045-7825(86)90079-4
        """
        self.Ei = parametrize(Ei)
        self.Fi = parametrize(Fi)
        self._init_damping(DEi, DFi)

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
            epsilon_rel_epsilon=np.tile(np.eye(6), (len(xi), 1, 1)),
            Csigma_epsilon=np.zeros((len(xi), 6, 6)),
            **self._prepare_damping(xi),
        )

    def potential(self, epsilon, epsilon0, prepare):
        d_epsilon = epsilon - epsilon0
        sig = prepare["EiFi"] * d_epsilon
        return 0.5 * np.sum(sig * d_epsilon, axis=1)

    def sigma(self, epsilon, epsilon0, epsilon_dot, prepare):
        d_epsilon = epsilon - epsilon0
        return prepare["EiFi"] * d_epsilon + self._sigma_damping(epsilon_dot, prepare)

    def sigma_epsilon(self, epsilon, epsilon0, epsilon_dot, prepare):
        return prepare["sigma_epsilon"]

    def epsilon_rel(self, epsilon, epsilon0, prepare):
        return epsilon - epsilon0

    def epsilon_rel_epsilon(self, epsilon, epsilon0, prepare):
        return prepare["epsilon_rel_epsilon"]

    def C(self, epsilon, epsilon0, prepare):
        return prepare["C_inv"]

    def Csigma_epsilon(self, epsilon, epsilon0, sigma, prepare):
        # derivative of C(epsilon) @ sigma w.r.t. epsilon, C is constant
        return prepare["Csigma_epsilon"]

    def epsilon_rel_inv(self, sigma, epsilon0, prepare):
        # solves C @ sigma - epsilon_rel(epsilon) = 0 for epsilon
        return epsilon0 + np.einsum("ijk,ik->ij", prepare["C_inv"], sigma)

    def C_inv(self, prepare):
        return prepare["C_inv"]


class Harsch2021(RodMaterialModel):
    def __init__(self, Ei, Fi):
        """
        Material model for shear deformable beam with nonlinear axial deformation.

        Parameters
        ----------
        Ei : np.ndarray (3,) or callable(xi) -> np.ndarray (3,)
            E0: extensional stiffness, i.e., stiffness in r_OP'-direction.
            E1: shear stiffness in e_y^K-direction.
            E2: shear stiffness in e_z^K-direction.
        Fi : np.ndarray (3,) or callable(xi) -> np.ndarray (3,)
            F0: torsional stiffness
            F1: flexural stiffness around e_y^K-direction.
            F2: flexural stiffness around e_z^K-direction.

        References
        ----------
        Harsch2021: https://doi.org/10.1177/10812865211000790
        """
        self.Ei = parametrize(Ei)
        self.Fi = parametrize(Fi)

    def prepare(self, xi):
        Ei = self.Ei(xi)
        Fi = self.Fi(xi)
        E0 = Ei[:, 0]
        C_n = np.apply_along_axis(np.diag, 1, Ei * np.array([0, 1, 1]))
        C_m = np.apply_along_axis(np.diag, 1, Fi)
        C_m_inv = np.apply_along_axis(np.diag, 1, 1 / Fi)
        # inverse of linearized n_Ga at the reference, i.e., diag(1 / Ei)
        C_n_inv = np.apply_along_axis(np.diag, 1, 1 / Ei)
        zeros = np.zeros((len(xi), 3, 3))

        return dict(
            E0=E0,
            C_n=C_n,
            C_n_inv=C_n_inv,
            C_m=C_m,
            C_m_inv=C_m_inv,
            zeros=zeros,
            eye6=np.tile(np.eye(6), (len(xi), 1, 1)),
        )

    def _n_Ga(self, epsilon, epsilon0, prepare):
        """n_Ga = C_n + a * I + b * Gamma x Gamma, with
        a = E0 * (1 - lambda0 / lambda) and b = E0 * lambda0 / lambda^3."""
        Ga = epsilon[:, :3]
        la = np.linalg.norm(Ga, axis=1)
        la0 = np.linalg.norm(epsilon0[:, :3], axis=1)
        E0 = prepare["E0"]
        a = E0 * (1 - la0 / la)
        b = E0 * la0 / la**3
        n_Ga = (
            prepare["C_n"]
            + a[:, None, None] * np.eye(3)
            + b[:, None, None] * np.einsum("ij,ik->ijk", Ga, Ga)
        )
        return n_Ga, a, b, la

    def potential(self, epsilon, epsilon0, prepare):
        d_epsilon = epsilon - epsilon0
        dG = d_epsilon[:, :3]
        dK = d_epsilon[:, 3:]
        la = np.linalg.norm(epsilon[:, :3], axis=1)
        la0 = np.linalg.norm(epsilon0[:, :3], axis=1)
        return 0.5 * (
            np.einsum("ij,ijk,ik->i", dG, prepare["C_n"], dG)
            + prepare["E0"] * (la - la0) ** 2
            + np.einsum("ij,ijk,ik->i", dK, prepare["C_m"], dK)
        )

    def sigma(self, epsilon, epsilon0, epsilon_dot, prepare):
        d_epsilon = epsilon - epsilon0
        la = np.linalg.norm(epsilon[:, :3], axis=1)
        la0 = np.linalg.norm(epsilon0[:, :3], axis=1)
        a = prepare["E0"] * (1 - la0 / la)
        sigma = np.empty_like(epsilon)
        sigma[:, :3] = (
            np.einsum("ijk,ik->ij", prepare["C_n"], d_epsilon[:, :3])
            + a[:, None] * epsilon[:, :3]
        )
        sigma[:, 3:] = np.einsum("ijk,ik->ij", prepare["C_m"], d_epsilon[:, 3:])
        return sigma

    def sigma_epsilon(self, epsilon, epsilon0, epsilon_dot, prepare):
        n_Ga = self._n_Ga(epsilon, epsilon0, prepare)[0]
        zeros = prepare["zeros"]
        return n_Ga, zeros, zeros, prepare["C_m"]

    # compliance form: C(epsilon) @ sigma - epsilon_rel(epsilon) = 0,
    # with C = sigma_epsilon^-1 and epsilon_rel = C @ sigma
    def C(self, epsilon, epsilon0, prepare):
        n_Ga = self._n_Ga(epsilon, epsilon0, prepare)[0]
        C = np.zeros((len(epsilon), 6, 6))
        C[:, :3, :3] = np.linalg.inv(n_Ga)
        C[:, 3:, 3:] = prepare["C_m_inv"]
        return C

    def epsilon_rel(self, epsilon, epsilon0, prepare):
        warn(
            "Harsch2021 is not convex, but its compliance form (la_e) requires "
            "an invertible sigma_epsilon: n_Ga becomes singular under compression, "
            "e.g., for lambda / lambda0 < E0 / (E0 + min(E1, E2)) without shear. "
            "Use it displacement-based or use a convex material model, e.g., Herrmann2026."
        )
        C = self.C(epsilon, epsilon0, prepare)
        sigma = self.sigma(epsilon, epsilon0, None, prepare)
        return np.einsum("ijk,ik->ij", C, sigma)

    def epsilon_rel_epsilon(self, epsilon, epsilon0, prepare):
        # d(C @ sigma)/d epsilon = C @ sigma_epsilon + d(C @ sigma)/d epsilon|_sigma
        sigma = self.sigma(epsilon, epsilon0, None, prepare)
        return prepare["eye6"] + self.Csigma_epsilon(epsilon, epsilon0, sigma, prepare)

    def Csigma_epsilon(self, epsilon, epsilon0, sigma, prepare):
        # d(n_Ga^-1 @ n)/d Gamma = -n_Ga^-1 @ T with T_k = d n_Ga/d Gamma_k @ v,
        # v = n_Ga^-1 @ n, and n_Ga as in self._n_Ga
        n_Ga, a, b, la = self._n_Ga(epsilon, epsilon0, prepare)
        n_Ga_inv = np.linalg.inv(n_Ga)
        Ga = epsilon[:, :3]
        v = np.einsum("ijk,ik->ij", n_Ga_inv, sigma[:, :3])
        Ga_v = np.sum(Ga * v, axis=1)
        T = b[:, None, None] * (
            np.einsum("ij,ik->ijk", v, Ga)
            + np.einsum("ij,ik->ijk", Ga, v)
            + Ga_v[:, None, None] * np.eye(3)
            - 3 * (Ga_v / la**2)[:, None, None] * np.einsum("ij,ik->ijk", Ga, Ga)
        )

        Csigma_epsilon = np.zeros((len(epsilon), 6, 6))
        Csigma_epsilon[:, :3, :3] = -n_Ga_inv @ T
        return Csigma_epsilon

    def epsilon_rel_inv(self, sigma, epsilon0, prepare, atol=1e-12, max_iter=20):
        # solves C(epsilon) @ sigma - epsilon_rel(epsilon) = 0 for epsilon,
        # which is equivalent to sigma(epsilon) - sigma = 0 as C is regular
        epsilon = epsilon0.copy()
        epsilon[:, :3] += np.einsum("ijk,ik->ij", prepare["C_n_inv"], sigma[:, :3])
        epsilon[:, 3:] += np.einsum("ijk,ik->ij", prepare["C_m_inv"], sigma[:, 3:])

        # Newton iteration for Gamma, starting at linearized solution
        for _ in range(max_iter):
            R = self.sigma(epsilon, epsilon0, None, prepare)[:, :3] - sigma[:, :3]
            if np.max(np.abs(R)) < atol:
                break
            n_Ga = self._n_Ga(epsilon, epsilon0, prepare)[0]
            epsilon[:, :3] -= np.linalg.solve(n_Ga, R[:, :, None])[:, :, 0]
        else:
            raise RuntimeError("epsilon_rel_inv: Newton iteration did not converge")

        return epsilon


class Herrmann2026(LinearViscousDamping, RodMaterialModel):
    def __init__(self, Ei, Fi, DEi=np.zeros(3), DFi=np.zeros(3), quadratic_part=0.5):
        self.Ei = parametrize(Ei)
        self.Fi = parametrize(Fi)

        assert (
            0 < quadratic_part <= 1
        ), f"quadratic part must be in (0, 1], but was {quadratic_part:.3f}"
        self.quadratic_part = quadratic_part

        self._init_damping(DEi, DFi)

    def prepare(self, xi):
        Ei = self.Ei(xi)
        Fi = self.Fi(xi)
        E0 = Ei[:, 0]
        # split of the axial stiffness into the ln part and quadratic part
        E_quad = self.quadratic_part * E0
        E_ln = (1 - self.quadratic_part) * E0
        has_ln = E_ln != 0.0

        prepare = {}
        prepare["E_ln"] = E_ln
        prepare["has_ln"] = has_ln
        # diagonal of the quadratic part, i.e., stored component-wise
        EiFi_quad = np.hstack([E_quad[:, None], Ei[:, 1:], Fi])
        prepare["EiFi_quad"] = EiFi_quad
        # constant parts of sigma_epsilon, only n_Ga[:, 0, 0] gets the ln part
        prepare["n_Ga_quad"] = np.apply_along_axis(np.diag, 1, EiFi_quad[:, :3])
        prepare["m_Ka"] = np.apply_along_axis(np.diag, 1, Fi)
        prepare["zeros"] = np.zeros((len(xi), 3, 3))
        prepare.update(self._prepare_damping(xi))
        return prepare

    def potential(self, epsilon, epsilon0, prepare):
        E_ln = prepare["E_ln"]
        has_ln = prepare["has_ln"]

        d_eps = epsilon - epsilon0
        d_epsx = d_eps[has_ln, 0]
        epsx_rel = epsilon[has_ln, 0] / epsilon0[has_ln, 0]

        epsx = epsilon[:, 0]
        epsx0 = epsilon0[has_ln, 0]
        Wx = np.zeros_like(epsx)
        Wx[has_ln] = -E_ln[has_ln] * epsx0 * (epsx0 * np.log(epsx_rel) - d_epsx)

        return Wx + 0.5 * np.sum(prepare["EiFi_quad"] * d_eps**2, axis=1)

    def sigma(self, epsilon, epsilon0, epsilon_dot, prepare):
        has_ln = prepare["has_ln"]
        epsx = epsilon[has_ln, 0]
        epsx0 = epsilon0[has_ln, 0]

        sigma = prepare["EiFi_quad"] * (epsilon - epsilon0)
        sigma[has_ln, 0] += prepare["E_ln"][has_ln] * epsx0 * (epsx - epsx0) / epsx
        return sigma + self._sigma_damping(epsilon_dot, prepare)

    def sigma_epsilon(self, epsilon, epsilon0, epsilon_dot, prepare):
        has_ln = prepare["has_ln"]
        epsx = epsilon[has_ln, 0]
        epsx0 = epsilon0[has_ln, 0]

        n_Ga = prepare["n_Ga_quad"].copy()
        n_Ga[has_ln, 0, 0] += prepare["E_ln"][has_ln] * epsx0**2 / epsx**2
        zeros = prepare["zeros"]
        return n_Ga, zeros, zeros, prepare["m_Ka"]

    # compliance form: C(epsilon) @ sigma - epsilon_rel(epsilon) = 0,
    # with C = sigma_epsilon^-1 and epsilon_rel = C @ sigma.
    # sigma_epsilon is diagonal, hence everything is evaluated component-wise.
    def _k(self, epsilon, epsilon0, prepare):
        """diagonal of sigma_epsilon"""
        has_ln = prepare["has_ln"]
        k = prepare["EiFi_quad"].copy()
        k[has_ln, 0] += (
            prepare["E_ln"][has_ln] * epsilon0[has_ln, 0] ** 2 / epsilon[has_ln, 0] ** 2
        )
        return k

    def C(self, epsilon, epsilon0, prepare):
        k = self._k(epsilon, epsilon0, prepare)
        C = np.zeros((len(epsilon), 6, 6))
        C[:, range(6), range(6)] = 1 / k
        return C

    def epsilon_rel(self, epsilon, epsilon0, prepare):
        k = self._k(epsilon, epsilon0, prepare)
        return self.sigma(epsilon, epsilon0, None, prepare) / k

    def epsilon_rel_epsilon(self, epsilon, epsilon0, prepare):
        # d(C @ sigma)/d epsilon = C @ sigma_epsilon + d(C @ sigma)/d epsilon|_sigma
        sigma = self.sigma(epsilon, epsilon0, None, prepare)
        epsilon_rel_epsilon = self.Csigma_epsilon(epsilon, epsilon0, sigma, prepare)
        epsilon_rel_epsilon[:, range(6), range(6)] += 1.0
        return epsilon_rel_epsilon

    def Csigma_epsilon(self, epsilon, epsilon0, sigma, prepare):
        # only C[0, 0] = 1 / k0 depends on epsilon, with
        # k0 = E_quad + E_ln * epsx0^2 / epsx^2 and dk0/depsx = -2 * E_ln * epsx0^2 / epsx^3
        has_ln = prepare["has_ln"]
        epsx = epsilon[has_ln, 0]
        epsx0 = epsilon0[has_ln, 0]
        k0 = self._k(epsilon, epsilon0, prepare)[:, 0]
        k0_epsx = np.zeros(len(epsilon))
        k0_epsx[has_ln] = -2 * prepare["E_ln"][has_ln] * epsx0**2 / epsx**3

        Csigma_epsilon = np.zeros((len(epsilon), 6, 6))
        Csigma_epsilon[:, 0, 0] = -sigma[:, 0] * k0_epsx / k0**2
        return Csigma_epsilon

    def epsilon_rel_inv(self, sigma, epsilon0, prepare):
        # solves C(epsilon) @ sigma - epsilon_rel(epsilon) = 0 for epsilon,
        # which is equivalent to sigma(epsilon) - sigma = 0 as C is regular
        epsilon = epsilon0 + sigma / prepare["EiFi_quad"]

        # axial part with ln: E_quad * (x - x0) + E_ln * x0 * (x - x0) / x = n,
        # multiplied by x: E_quad * x^2 + b * x - E_ln * x0^2 = 0, with
        # b = (E_ln - E_quad) * x0 - n, which has exactly one positive root.
        # Without ln, the linear solution from above is already correct.
        has_ln = prepare["has_ln"]
        x0 = epsilon0[has_ln, 0]
        E_quad = prepare["EiFi_quad"][has_ln, 0]
        E_ln = prepare["E_ln"][has_ln]
        b = (E_ln - E_quad) * x0 - sigma[has_ln, 0]
        sqrt_D = np.sqrt(b**2 + 4 * E_quad * E_ln * x0**2)
        # cancellation free root
        epsilon[has_ln, 0] = np.where(
            b > 0,
            2 * E_ln * x0**2 / (b + sqrt_D),
            (-b + sqrt_D) / (2 * E_quad),
        )
        return epsilon


class Herrmann2026_lnexp(LinearViscousDamping, RodMaterialModel):
    def __init__(
        self, Ei, Fi, DEi=np.zeros(3), DFi=np.zeros(3), ln_part=0.25, exp_part=0.25
    ):
        """
        Material model with quadratic shear, torsion and bending and an axial
        part, which is a combination of a quadratic, a ln and an exp part.
        With x = Gamma_1, x0 = Gamma0_1 and d = (x - x0) / x0, the axial
        potential reads

            W_x = 0.5 * E_quad * (x - x0)^2
                  + E_ln * x0^2 * (d - ln(1 + d))
                  + E_exp * x0^2 * (exp(d) - 1 - d),

        where E_ln = ln_part * E0, E_exp = exp_part * E0 and E_quad is the
        remaining part of E0. All parts have zero stress at the reference and
        add up to the axial stiffness E0 there. The ln part stiffens under
        compression (x -> 0), the exp part under extension.

        Parameters
        ----------
        Ei : np.ndarray (3,) or callable(xi) -> np.ndarray (3,)
            E0: axial stiffness, E1, E2: shear stiffnesses.
        Fi : np.ndarray (3,) or callable(xi) -> np.ndarray (3,)
            F0: torsional stiffness, F1, F2: flexural stiffnesses.
        DEi, DFi : np.ndarray (3,) or callable(xi) -> np.ndarray (3,)
            Linear viscous damping, see LinearViscousDamping.
        ln_part, exp_part : float
            Parts of E0 for the ln and the exp part.
        """
        self.Ei = parametrize(Ei)
        self.Fi = parametrize(Fi)

        assert ln_part >= 0 and exp_part >= 0, "ln_part and exp_part must be >= 0"
        assert ln_part + exp_part <= 1, "ln_part + exp_part must be <= 1"
        # the ln part alone limits the axial force in extension, the exp part
        # alone in compression
        assert ln_part < 1, "a pure ln part is bounded in extension"
        assert exp_part < 1, "a pure exp part is bounded in compression"
        self.ln_part = ln_part
        self.exp_part = exp_part

        self._init_damping(DEi, DFi)

    def prepare(self, xi):
        Ei = self.Ei(xi)
        Fi = self.Fi(xi)
        E0 = Ei[:, 0]
        # split of the axial stiffness into the quadratic, ln and exp part
        E_ln = self.ln_part * E0
        E_exp = self.exp_part * E0
        E_quad = E0 - E_ln - E_exp

        prepare = {}
        prepare["E_ln"] = E_ln
        prepare["E_exp"] = E_exp
        prepare["has_ln"] = E_ln != 0.0
        prepare["has_exp"] = E_exp != 0.0
        # diagonal of the quadratic part, i.e., stored component-wise
        EiFi_quad = np.hstack([E_quad[:, None], Ei[:, 1:], Fi])
        prepare["EiFi_quad"] = EiFi_quad
        # constant parts of sigma_epsilon, only n_Ga[:, 0, 0] gets the ln and exp part
        prepare["n_Ga_quad"] = np.apply_along_axis(np.diag, 1, EiFi_quad[:, :3])
        prepare["m_Ka"] = np.apply_along_axis(np.diag, 1, Fi)
        prepare["zeros"] = np.zeros((len(xi), 3, 3))
        prepare.update(self._prepare_damping(xi))
        return prepare

    def potential(self, epsilon, epsilon0, prepare):
        d_eps = epsilon - epsilon0
        W = 0.5 * np.sum(prepare["EiFi_quad"] * d_eps**2, axis=1)

        h = prepare["has_ln"]
        x, x0 = epsilon[h, 0], epsilon0[h, 0]
        W[h] += -prepare["E_ln"][h] * x0 * (x0 * np.log(x / x0) - (x - x0))

        h = prepare["has_exp"]
        x0 = epsilon0[h, 0]
        d = (epsilon[h, 0] - x0) / x0
        W[h] += prepare["E_exp"][h] * x0**2 * (np.exp(d) - 1 - d)
        return W

    def sigma(self, epsilon, epsilon0, epsilon_dot, prepare):
        sigma = prepare["EiFi_quad"] * (epsilon - epsilon0)

        h = prepare["has_ln"]
        x, x0 = epsilon[h, 0], epsilon0[h, 0]
        sigma[h, 0] += prepare["E_ln"][h] * x0 * (x - x0) / x

        h = prepare["has_exp"]
        x0 = epsilon0[h, 0]
        d = (epsilon[h, 0] - x0) / x0
        sigma[h, 0] += prepare["E_exp"][h] * x0 * (np.exp(d) - 1)
        return sigma + self._sigma_damping(epsilon_dot, prepare)

    def sigma_epsilon(self, epsilon, epsilon0, epsilon_dot, prepare):
        n_Ga = prepare["n_Ga_quad"].copy()
        n_Ga[:, 0, 0] = self._k(epsilon, epsilon0, prepare)[:, 0]
        zeros = prepare["zeros"]
        return n_Ga, zeros, zeros, prepare["m_Ka"]

    # compliance form: C(epsilon) @ sigma - epsilon_rel(epsilon) = 0,
    # with C = sigma_epsilon^-1 and epsilon_rel = C @ sigma.
    # sigma_epsilon is diagonal, hence everything is evaluated component-wise.
    def _k(self, epsilon, epsilon0, prepare):
        """diagonal of sigma_epsilon"""
        k = prepare["EiFi_quad"].copy()

        h = prepare["has_ln"]
        x, x0 = epsilon[h, 0], epsilon0[h, 0]
        k[h, 0] += prepare["E_ln"][h] * x0**2 / x**2

        h = prepare["has_exp"]
        x0 = epsilon0[h, 0]
        k[h, 0] += prepare["E_exp"][h] * np.exp((epsilon[h, 0] - x0) / x0)
        return k

    def C(self, epsilon, epsilon0, prepare):
        k = self._k(epsilon, epsilon0, prepare)
        C = np.zeros((len(epsilon), 6, 6))
        C[:, range(6), range(6)] = 1 / k
        return C

    def epsilon_rel(self, epsilon, epsilon0, prepare):
        k = self._k(epsilon, epsilon0, prepare)
        return self.sigma(epsilon, epsilon0, None, prepare) / k

    def epsilon_rel_epsilon(self, epsilon, epsilon0, prepare):
        # d(C @ sigma)/d epsilon = C @ sigma_epsilon + d(C @ sigma)/d epsilon|_sigma
        sigma = self.sigma(epsilon, epsilon0, None, prepare)
        epsilon_rel_epsilon = self.Csigma_epsilon(epsilon, epsilon0, sigma, prepare)
        epsilon_rel_epsilon[:, range(6), range(6)] += 1.0
        return epsilon_rel_epsilon

    def Csigma_epsilon(self, epsilon, epsilon0, sigma, prepare):
        # only C[0, 0] = 1 / k0 depends on epsilon, with dk0/dx from the ln
        # part -2 * E_ln * x0^2 / x^3 and from the exp part E_exp * exp(d) / x0
        k0 = self._k(epsilon, epsilon0, prepare)[:, 0]
        k0_x = np.zeros(len(epsilon))

        h = prepare["has_ln"]
        x, x0 = epsilon[h, 0], epsilon0[h, 0]
        k0_x[h] += -2 * prepare["E_ln"][h] * x0**2 / x**3

        h = prepare["has_exp"]
        x0 = epsilon0[h, 0]
        k0_x[h] += prepare["E_exp"][h] * np.exp((epsilon[h, 0] - x0) / x0) / x0

        Csigma_epsilon = np.zeros((len(epsilon), 6, 6))
        Csigma_epsilon[:, 0, 0] = -sigma[:, 0] * k0_x / k0**2
        return Csigma_epsilon

    def epsilon_rel_inv(self, sigma, epsilon0, prepare, rtol=1e-12, max_iter=100):
        # solves C(epsilon) @ sigma - epsilon_rel(epsilon) = 0 for epsilon,
        # which is equivalent to sigma(epsilon) - sigma = 0 as C is regular
        epsilon = epsilon0 + sigma / prepare["EiFi_quad"]

        # axial part: no closed form solution, hence a Newton iteration on
        # n(x) - sigma_x = 0, safeguarded by bisection, as n(x) is monotonically
        # increasing, but neither convex nor concave
        n = sigma[:, 0]
        x0 = epsilon0[:, 0]
        has_ln = prepare["has_ln"]

        def n_k(x):
            eps = epsilon0.copy()
            eps[:, 0] = x
            n_x = self.sigma(eps, epsilon0, None, prepare)[:, 0]
            k_x = self._k(eps, epsilon0, prepare)[:, 0]
            return n_x, k_x

        # bracket [lo, hi] with n(lo) <= n <= n(hi); the ln part requires x > 0
        lo, hi = x0.copy(), x0.copy()
        dist = x0.copy()
        for _ in range(max_iter):
            too_high = n_k(lo)[0] > n
            if not np.any(too_high):
                break
            lo[too_high] = np.where(has_ln, 0.5 * lo, x0 - dist)[too_high]
            dist[too_high] *= 2
        dist = x0.copy()
        for _ in range(max_iter):
            too_low = n_k(hi)[0] < n
            if not np.any(too_low):
                break
            hi[too_low] = (x0 + dist)[too_low]
            dist[too_low] *= 2

        x = np.clip(epsilon[:, 0], lo, hi)
        for _ in range(max_iter):
            n_x, k_x = n_k(x)
            R = n_x - n
            if np.max(np.abs(R) / (1 + np.abs(n))) < rtol:
                break
            lo = np.where(R < 0, x, lo)
            hi = np.where(R > 0, x, hi)
            x_new = x - R / k_x
            outside = (x_new <= lo) | (x_new >= hi)
            x = np.where(outside, 0.5 * (lo + hi), x_new)
        else:
            raise RuntimeError("epsilon_rel_inv: Newton iteration did not converge")

        epsilon[:, 0] = x
        return epsilon
