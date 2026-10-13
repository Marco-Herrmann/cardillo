# SPDX-License-Identifier: MIT
# Copyright (c) 2011–2026 Joris J.C. Remmers

"""Linear viscoelastic material using a generalized Maxwell model.

Zero-transverse-stress reduction for rod kinematics. The active stress/strain
components are [11, 13, 12], with engineering shear strains. The public
interface uses six-component Voigt arrays [11, 22, 33, 23, 13, 12].

All Maxwell branches share a constant Poisson's ratio. The equilibrium spring
is updated incrementally, so total strain is not needed.
"""

from .BaseMaterial import BaseMaterial
from numpy import zeros, diag, exp, expm1, array, asarray, ix_, isclose
from typing import Tuple


class ViscoElasticity(BaseMaterial):
    """Generalized Maxwell viscoelasticity with zero transverse stresses.

    Parameters
    ----------
    E : float
        Instantaneous Young's modulus.
    nu : float
        Constant Poisson's ratio.
    Einf : float, optional
        Equilibrium Young's modulus (default 0.1*E).
    nMaxwell : int, optional
        Number of Maxwell branches (default 3).
    relaxTimes : sequence of float, optional
        Relaxation times.
    relaxModuli : sequence of float, optional
        Branch Young's moduli; their sum should be E - Einf.

    Notes
    -----
    getStress expects strain increments relative to the last committed state.
    The material's BaseMaterial history mechanism must keep trial states
    separate from committed states until commit() is called.
    """

    def initializeMaterialPoint(self, elementID, intpointID):
        if not self.hasHistory(elementID, intpointID, "sigma"):
            self.oldHistory[(elementID, intpointID, "sigma")] = zeros(6)
            for i in range(self.nMaxwell):
                self.oldHistory[(elementID, intpointID, f"sigma_i_{i}")] = zeros(3)

    def __init__(
        self, E, nu, Einf=None, nMaxwell=3, relaxTimes=None, relaxModuli=None
    ) -> None:
        super().__init__()

        self.E = E
        self.nu = nu
        self.nMaxwell = nMaxwell

        if self.nMaxwell < 0:
            raise ValueError("nMaxwell must be nonnegative")

        # Set default values for viscoelastic parameters
        self.Einf = 0.1 * self.E if Einf is None else Einf

        # Calculate relaxation moduli if not provided
        if relaxModuli is None:
            E_relax_total = self.E - self.Einf
            self.relaxModuli = (
                [E_relax_total / self.nMaxwell] * self.nMaxwell if self.nMaxwell else []
            )
        else:
            self.relaxModuli = relaxModuli

        # Calculate relaxation times if not provided
        if relaxTimes is None:
            import numpy as np

            self.relaxTimes = list(np.logspace(-1, 2, self.nMaxwell))
        else:
            self.relaxTimes = relaxTimes

        # Validate input
        if self.nMaxwell < 0:
            raise ValueError("nMaxwell must be nonnegative")
        if len(self.relaxModuli) != self.nMaxwell:
            raise ValueError("Length of relaxModuli must equal nMaxwell")
        if len(self.relaxTimes) != self.nMaxwell:
            raise ValueError("Length of relaxTimes must equal nMaxwell")
        if not (-1.0 < self.nu < 0.5):
            raise ValueError("nu must satisfy -1 < nu < 0.5")
        if self.E <= 0 or self.Einf < 0:
            raise ValueError("E must be positive and Einf nonnegative")
        if any(tau <= 0 for tau in self.relaxTimes):
            raise ValueError("All relaxation times must be positive")
        if any(mod < 0 for mod in self.relaxModuli):
            raise ValueError("All relaxation moduli must be nonnegative")
        if not isclose(
            sum(self.relaxModuli) + self.Einf, self.E, rtol=1e-8, atol=1e-10 * self.E
        ):
            raise ValueError("Einf + sum(relaxModuli) must equal E")

        # Reduced equilibrium elastic stiffness: [11, 13, 12]
        self.Cinf = diag(
            [
                self.Einf,
                self.Einf / (2.0 * (1.0 + self.nu)),
                self.Einf / (2.0 * (1.0 + self.nu)),
            ]
        )

        # Reduced Maxwell branch stiffness matrices
        self.Cmaxwell = [
            diag([E_i, E_i / (2.0 * (1.0 + self.nu)), E_i / (2.0 * (1.0 + self.nu))])
            for E_i in self.relaxModuli
        ]

    def getStress(
        self, kinematics, actions, element, intpointID
    ) -> Tuple[array, array]:
        """Return stress and consistent tangent in six-component Voigt form.

        Input dstrain may have 6 components [11,22,33,23,13,12], or
        3 reduced components [11,13,12]. Transverse components of a
        six-component input are ignored by the reduced constitutive model.
        """
        elementID = element.id
        dtime = actions.dtime
        if dtime < 0:
            raise ValueError("Time increment must be nonnegative")

        self.initializeMaterialPoint(elementID, intpointID)

        # Get committed total stress (copy to avoid changing history in place)
        sigma = self.getHistoryParameter(elementID, intpointID, "sigma").copy()

        # Convert strain increment to the reduced three-component representation
        dstrain_input = asarray(kinematics.dstrain)
        active = [0, 4, 5]
        if len(dstrain_input) == 6:
            dstrain = dstrain_input[active]
        elif len(dstrain_input) == 3:
            dstrain = dstrain_input
        else:
            raise ValueError("dstrain must contain 3 or 6 components")

        # Equilibrium spring: incremental update (no total strain needed)
        sigma[active] += self.Cinf @ dstrain

        # Consistent reduced tangent, starting with equilibrium spring
        tang = self.Cinf.copy()

        # Update each Maxwell branch using an exact constant-strain-rate step
        for i in range(self.nMaxwell):
            sigma_i = self.getHistoryParameter(
                elementID, intpointID, f"sigma_i_{i}"
            ).copy()
            tau_i = self.relaxTimes[i]
            x = dtime / tau_i
            a_i = exp(-x)
            # b_i = (1-exp(-x))/x, with stable limit b_i -> 1 at x -> 0
            b_i = -expm1(-x) / x if x > 0 else 1.0

            sigma_i_new = a_i * sigma_i + b_i * (self.Cmaxwell[i] @ dstrain)

            # Increment total stress by the change in branch stress
            sigma[active] += sigma_i_new - sigma_i

            # Derivative of updated branch stress w.r.t. strain increment
            tang += b_i * self.Cmaxwell[i]

            # Store trial branch stress; BaseMaterial.commit() accepts it later
            self.setHistoryParameter(elementID, intpointID, f"sigma_i_{i}", sigma_i_new)

        # All transverse stresses are zero by construction
        sigma[[1, 2, 3]] = 0.0
        self.setHistoryParameter(elementID, intpointID, "sigma", sigma)

        # Embed reduced tangent into the original six-component Voigt format
        tang6 = zeros((6, 6))
        tang6[ix_(active, active)] = tang
        return sigma, tang6
