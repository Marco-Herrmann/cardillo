# SPDX-License-Identifier: MIT
# Adapted from the viscoplastic material model by Joris J.C. Remmers (2011–2026).
"""Perzyna viscoplasticity with linear isotropic hardening and zero transverse stresses.

Active Voigt components: (11, 13, 12), with engineering shear strains.
The six-component interface uses standard Voigt order (11,22,33,23,13,12).
"""

from .BaseMaterial import BaseMaterial
from numpy import zeros, dot, array, diag, sqrt
from typing import Tuple


class ViscoPlasticity(BaseMaterial):
    """Reduced-stress Perzyna model, preserving the original material interface."""

    def initializeMaterialPoint(self, elementID, intpointID):
        if not self.hasHistory(elementID, intpointID, "sigma"):
            self.oldHistory[(elementID, intpointID, "sigma")] = zeros(6)
            self.oldHistory[(elementID, intpointID, "eelas")] = zeros(6)
            self.oldHistory[(elementID, intpointID, "eplas")] = zeros(6)
            self.oldHistory[(elementID, intpointID, "eqplas")] = 0.0

    def __init__(self, E, nu, syield, gamma, n=1.0, hard=0.0):
        super().__init__()
        if E <= 0 or not (-1.0 < nu < 0.5):
            raise ValueError("Require E > 0 and -1 < nu < 0.5")
        if syield <= 0 or gamma < 0 or n <= 0 or hard < 0:
            raise ValueError("Require syield > 0, gamma >= 0, n > 0, hard >= 0")

        self.E = E
        self.nu = nu
        self.syield = syield
        self.gamma = gamma
        self.n = n
        self.hard = hard

        # Reduced elastic stiffness: (11, 13, 12).
        self.eg = E / (2.0 * (1.0 + nu))
        self.active = array([0, 4, 5])
        self.weights = array([1.0, 3.0, 3.0])
        self.C = diag([E, self.eg, self.eg])
        self.A = diag(self.C) * self.weights

        # Embed the reduced stiffness into the original 6x6 Voigt layout.
        self.ctang = zeros((6, 6))
        self.ctang[self.active[:, None], self.active] = self.C

        self.tolerance = 1.0e-13
        self.maxIter = 100  # Bisection: robust even for fractional n.

    def getStress(
        self, kinematics, actions, element, intpointID
    ) -> Tuple[array, array]:
        """Return stress and consistent tangent, with zero transverse stresses.

        dstrain is measured relative to the last committed material state.
        Both 3-component (11,13,12) and 6-component Voigt inputs are accepted.
        Output has the same number of components as input.
        """
        elementID = element.id
        dtime = actions.dtime
        self.initializeMaterialPoint(elementID, intpointID)

        # Retrieve history; copies prevent modifications to committed arrays.
        eelas = self.getHistoryParameter(elementID, intpointID, "eelas").copy()
        eplas = self.getHistoryParameter(elementID, intpointID, "eplas").copy()
        eqplas = self.getHistoryParameter(elementID, intpointID, "eqplas")
        sigma = self.getHistoryParameter(elementID, intpointID, "sigma").copy()

        # Extract the retained components of the strain increment.
        if len(kinematics.dstrain) == 6:
            dstrain = array(kinematics.dstrain)[self.active]
        elif len(kinematics.dstrain) == 3:
            dstrain = array(kinematics.dstrain)
        else:
            raise ValueError("dstrain must have 3 or 6 components")

        # Elastic predictor.
        sigma_trial = sigma[self.active] + dot(self.C, dstrain)
        smises = sqrt(dot(self.weights * sigma_trial, sigma_trial))
        syield_current = self.syield + self.hard * eqplas
        tang = self.C.copy()
        sigma_new = sigma_trial.copy()
        deqpl = 0.0

        # Check for viscoplastic loading.
        if smises > syield_current and dtime > 0 and self.gamma > 0:
            k = self.gamma * dtime

            # Corrected stress for a given t = deqpl / sigma_VM_new.
            def response(t):
                denominator = 1.0 + t * self.A
                stress = sigma_trial / denominator
                equivalent = sqrt(dot(self.weights * stress, stress))
                plastic_increment = t * equivalent
                yield_stress = self.syield + self.hard * (eqplas + plastic_increment)
                return stress, equivalent, plastic_increment, yield_stress, denominator

            # Solve: sigma_VM = sigma_Y * [1 + (deqpl / (gamma*dt))**(1/n)].
            # At t=0 residual is positive; at sufficiently large t it is negative.
            def residual(t):
                _, equivalent, plastic_increment, yield_stress, _ = response(t)
                return equivalent - yield_stress * (
                    1.0 + (plastic_increment / k) ** (1.0 / self.n)
                )

            lo, hi = 0.0, 1.0 / max(self.A)
            while residual(hi) > 0:
                hi *= 2.0
                if hi > 1e100:
                    raise RuntimeError("Unable to bracket Perzyna plastic correction")

            converged = False
            for iteration in range(self.maxIter):
                t = 0.5 * (lo + hi)
                r = residual(t)
                if abs(r) <= self.tolerance * max(self.syield, smises):
                    converged = True
                    break
                if r > 0:
                    lo = t
                else:
                    hi = t
            if not converged:
                raise RuntimeError("Viscoplastic correction did not converge")

            sigma_new, equivalent, deqpl, syield_iter, denominator = response(t)

            # Plastic strain increment: deqplas * W*sigma/sigma_VM.
            flow = self.weights * sigma_new / equivalent
            eplas[self.active] += deqpl * flow
            eelas[self.active] += dstrain - deqpl * flow
            eqplas += deqpl

            # Consistent algorithmic tangent: differentiate the scalar equation.
            # s(t) = sqrt(sum_i W_i * (sigma_trial_i/(1+t*A_i))**2).
            ds_dtrial = self.weights * sigma_new / (equivalent * denominator)
            ds_dt = -dot(self.weights * sigma_new**2, self.A / denominator) / equivalent
            q = (deqpl / k) ** (1.0 / self.n)
            # d/dp [Y(p)*(1+(p/k)^(1/n))].
            D = self.hard * (1.0 + q) + syield_iter * q / (self.n * deqpl)
            dF_dt = ds_dt - D * (equivalent + t * ds_dt)
            dF_dtrial = ds_dtrial * (1.0 - D * t)
            dt_dtrial = -dF_dtrial / dF_dt
            correction = self.A * sigma_trial / denominator**2
            jac_stress_trial = (
                diag(1.0 / denominator) - correction[:, None] * dt_dtrial[None, :]
            )
            tang = dot(jac_stress_trial, self.C)

        else:
            # Elastic response, including increments with dtime == 0.
            eelas[self.active] += dstrain

        # Store updated history, preserving zero transverse stresses.
        sigma[:] = 0.0
        sigma[self.active] = sigma_new

        self.setHistoryParameter(elementID, intpointID, "eelas", eelas)
        self.setHistoryParameter(elementID, intpointID, "eplas", eplas)
        self.setHistoryParameter(elementID, intpointID, "sigma", sigma)
        self.setHistoryParameter(elementID, intpointID, "eqplas", eqplas)

        # Embed tangent into the original Voigt layout.
        tang6 = zeros((6, 6))
        tang6[self.active[:, None], self.active] = tang

        if len(kinematics.dstrain) == 6:
            return sigma, tang6
        return sigma[self.active], tang
