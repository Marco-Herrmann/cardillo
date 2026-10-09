# SPDX-License-Identifier: MIT
# Copyright (c) 2011–2026 Joris J.C. Remmers

"""
Viscoelastic material model using generalized Maxwell model.

This module implements a linear viscoelastic material based on the generalized
Maxwell model (Prony series representation). The material behavior is modeled
as a spring in parallel with multiple Maxwell elements (spring-dashpot pairs).
"""

from .BaseMaterial import BaseMaterial
from .MatUtils import transform3To2, transform2To3
from numpy import zeros, dot, exp, array
from typing import Tuple, Union


class ViscoElasticity(BaseMaterial):
    """
    Viscoelastic material model using generalized Maxwell model.

    This class implements a linear viscoelastic constitutive model based on
    the generalized Maxwell model (Prony series). The material consists of
    an elastic spring in parallel with N Maxwell elements, each having a
    relaxation time and modulus.

    The relaxation modulus is expressed as:
    E(t) = E_inf + sum_i E_i * exp(-t/tau_i)

    where:
    - E_inf: long-term (equilibrium) modulus
    - E_i: modulus of Maxwell element i
    - tau_i: relaxation time of Maxwell element i

    Required Properties
    -------------------
    E : float
        Instantaneous Young's modulus (total modulus at t=0).
    nu : float
        Poisson's ratio (assumed constant).
    Einf : float, optional
        Long-term equilibrium modulus. Default: 0.1*E
    nMaxwell : int, optional
        Number of Maxwell elements. Default: 3
    relaxTimes : list of float, optional
        Relaxation times for each Maxwell element [tau_1, tau_2, ...].
        If not provided, logarithmically spaced times are used.
    relaxModuli : list of float, optional
        Relaxation moduli for each Maxwell element [E_1, E_2, ...].
        If not provided, equal distribution is used.

    Examples
    --------
    Material properties in input file:

    <Material>
        type = "ViscoElasticity";
        E = 1000.0;
        nu = 0.3;
        Einf = 100.0;
        nMaxwell = 2;
        relaxTimes = [1.0, 10.0];
        relaxModuli = [300.0, 600.0];
    </Material>
    """

    def initializeMaterialPoint(self, elementID, intpointID):

        if not self.hasHistory(elementID, intpointID, "sigma"):
            self.oldHistory[(elementID, intpointID, "sigma")] = zeros(6)

            for i in range(self.nMaxwell):
                self.oldHistory[(elementID, intpointID, f"eps_i_{i}")] = zeros(6)

    def __init__(
        self, E, nu, Einf=None, nMaxwell=3, relaxTimes=None, relaxModuli=None
    ) -> None:
        """
        Initialize the viscoelastic material model.

        Parameters
        ----------
        E: Young's modulus (mandatory)
        nu: poisson's ratio (mandatory)
        Einf: long term (equilibrium) Young's modulus
        nMaxwell: number of Maxwell elements
        relaxTimes: relax times for the Maxwell elements
        relaxModuli: relax moduli for the Maxwell elements
        """
        super().__init__()

        self.E = E
        self.nu = nu
        self.nMaxwell = nMaxwell

        # Set default values for viscoelastic parameters
        if Einf is None:
            self.Einf = 0.1 * self.E
        else:
            self.Einf = Einf

        # Calculate relaxation moduli if not provided
        if relaxModuli is None:
            # Equal distribution of relaxation moduli
            E_relax_total = self.E - self.Einf
            self.relaxModuli = [E_relax_total / self.nMaxwell] * self.nMaxwell
        else:
            self.relaxModuli = relaxModuli

        # Calculate relaxation times if not provided
        if relaxTimes is None:
            # Logarithmically spaced relaxation times
            import numpy as np

            self.relaxTimes = list(np.logspace(-1, 2, self.nMaxwell))
        else:
            self.relaxTimes = relaxTimes

        # Validate input
        if len(self.relaxModuli) != self.nMaxwell:
            raise ValueError(
                f"Length of relaxModuli ({len(self.relaxModuli)}) must equal nMaxwell ({self.nMaxwell})"
            )

        if len(self.relaxTimes) != self.nMaxwell:
            raise ValueError(
                f"Length of relaxTimes ({len(self.relaxTimes)}) must equal nMaxwell ({self.nMaxwell})"
            )

        # Compute elastic constants
        self.ebulk3 = self.Einf / (1.0 - 2.0 * self.nu)
        self.eg2 = self.Einf / (1.0 + self.nu)
        self.eg = 0.5 * self.eg2
        self.elam = (self.ebulk3 - self.eg2) / 3.0

        # Construct long-term elastic stiffness matrix
        self.Cinf = zeros(shape=(6, 6))
        self.Cinf[:3, :3] = self.elam

        self.Cinf[0, 0] += self.eg2
        self.Cinf[1, 1] = self.Cinf[0, 0]
        self.Cinf[2, 2] = self.Cinf[0, 0]

        self.Cinf[3, 3] = self.eg
        self.Cinf[4, 4] = self.Cinf[3, 3]
        self.Cinf[5, 5] = self.Cinf[3, 3]

    def getStress(
        self, kinematics, actions, element, intpointID
    ) -> Tuple[array, array]:
        """
        Compute stress and tangent stiffness for the viscoelastic model.

        Uses a generalized Maxwell model with multiple relaxation times to
        compute the stress from the strain history. The algorithm integrates
        the evolution equations for the internal strains of each Maxwell element
        using an exponential time integration scheme.

        Parameters
        ----------
        kinematics : Kinematics
            Kinematics object containing strain tensor and time step information.
            Required attributes:
            - dstrain: strain increment (3 or 6 components)
            - strain: total strain (optional, for debugging)

        Returns
        -------
        sigma : ndarray
            Stress tensor (3 or 6 components, depending on input).
        tang : ndarray
            Tangent stiffness matrix (3x3 or 6x6).

        Notes
        -----
        The stress is computed as:
        sigma = C_inf : (epsilon - sum_i eps_i)

        where eps_i are the internal strains of each Maxwell element, evolved as:
        d(eps_i)/dt = (epsilon - eps_i) / tau_i
        """
        elementID = element.id
        dtime = actions.dtime

        self.initializeMaterialPoint(elementID, intpointID)

        # Get history variables
        sigma = self.getHistoryParameter(elementID, intpointID, "sigma")

        # Convert strain to 6 components if needed
        if len(kinematics.dstrain) == 6:
            dstrain = kinematics.dstrain
        else:
            dstrain = transform2To3(kinematics.dstrain)

        # Initialize stress with long-term elastic response
        sigma_elastic = dot(self.Cinf, dstrain)
        sigma += sigma_elastic

        # Initialize tangent with long-term elastic stiffness
        tang = self.Cinf.copy()

        # Update internal strains for each Maxwell element
        if dtime > 0:
            for i in range(self.nMaxwell):
                eps_i = self.getHistoryParameter(elementID, intpointID, f"eps_i_{i}")
                tau_i = self.relaxTimes[i]
                E_i = self.relaxModuli[i]

                # Compute relaxation factor
                exp_factor = exp(-dtime / tau_i)

                # Update internal strain using exponential integrator
                # eps_i^{n+1} = exp(-dt/tau) * eps_i^n + (1 - exp(-dt/tau)) * epsilon^{n+1}
                deps_i = exp_factor * eps_i + (1.0 - exp_factor) * dstrain

                # Compute contribution to stress
                # sigma += E_i/Einf * C_inf : (epsilon - eps_i)
                factor = E_i / self.Einf
                stress_contribution = factor * dot(self.Cinf, dstrain - deps_i)
                sigma += stress_contribution

                # Update tangent: add contribution from this Maxwell element
                # C_eff = C_inf * (1 + sum_i E_i/Einf * (1 - exp(-dt/tau_i)))
                tang_factor = factor * (1.0 - exp_factor)
                tang += tang_factor * self.Cinf

                # Store updated internal strain
                self.setHistoryParameter(elementID, intpointID, f"eps_i_{i}", deps_i)

        # Update history
        self.setHistoryParameter(elementID, intpointID, "sigma", sigma)

        # Return stress and tangent in appropriate format
        if len(kinematics.dstrain) == 6:
            return sigma, tang
        else:
            return transform3To2(sigma, tang)
