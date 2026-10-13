# SPDX-License-Identifier: MIT
# Copyright (c) 2011–2026 Joris J.C. Remmers

from .BaseMaterial import BaseMaterial
from numpy import zeros, dot, array, sqrt, diag, outer, ix_, eye, asarray


class Plasticity(BaseMaterial):
    """
    Rate-independent plasticity model with linear kinematic hardening.

    Reduced constitutive behavior for a beam cross-section with
    sigma22 = sigma33 = sigma23 = 0. The active Voigt components are
    [11, 13, 12] (indices [0, 4, 5]); shear strains are engineering strains.

    The interface retains six-component stresses and 6x6 tangents.
    """

    def initializeMaterialPoint(self, elementID, intpointID):

        if not self.hasHistory(elementID, intpointID, "sigma"):
            self.oldHistory[(elementID, intpointID, "sigma")] = zeros(6)
            self.oldHistory[(elementID, intpointID, "eelas")] = zeros(6)
            self.oldHistory[(elementID, intpointID, "eplas")] = zeros(6)
            self.oldHistory[(elementID, intpointID, "alpha")] = zeros(6)

    def __init__(self, E, nu, syield, hard):

        super().__init__()

        self.E = E
        self.nu = nu
        self.syield = syield
        self.hard = hard

        self.tolerance = 1.0e-6

        self.eg = self.E / (2.0 * (1.0 + self.nu))

        # Reduced stiffness embedded in six-component Voigt notation.
        self.ctang = zeros(shape=(6, 6))
        self.ctang[0, 0] = self.E
        self.ctang[4, 4] = self.eg
        self.ctang[5, 5] = self.eg

        self.active = array([0, 4, 5])
        self.weights = array([1.0, 3.0, 3.0])

        # Effective reduced backstress:
        # alpha_hat = [alpha11-alpha22, alpha13, alpha12].
        # The 3D rule alpha += hard * flow * dp yields
        # d(alpha11-alpha22) = 1.5*hard*eta11/seq*dp.
        self.hard_weights = array([1.5, 1.0, 1.0])

    # ------------------------------------------------------------------------------
    #  pre:  kinematics object containing strain increment (kinematics.dstrain)
    #  post: stress vector and tangent matrix
    # ------------------------------------------------------------------------------

    def getStress(self, kinematics, actions, element, intpointID):

        elementID = element.id

        self.initializeMaterialPoint(elementID, intpointID)

        # Copy history so trial evaluations cannot mutate committed state.
        eelas = self.getHistoryParameter(elementID, intpointID, "eelas").copy()
        eplas = self.getHistoryParameter(elementID, intpointID, "eplas").copy()
        alpha = self.getHistoryParameter(elementID, intpointID, "alpha").copy()
        sigma = self.getHistoryParameter(elementID, intpointID, "sigma").copy()

        # A six-component strain vector is expected by the beam interface.
        # Three-component input is also accepted in [11,13,12] order.
        if len(kinematics.dstrain) == 6:
            dstrain = asarray(kinematics.dstrain)
        elif len(kinematics.dstrain) == 3:
            dstrain = zeros(6)
            dstrain[self.active] = kinematics.dstrain
        else:
            raise ValueError("dstrain must have 3 or 6 components")

        eelas += dstrain * array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        sigma += dot(self.ctang, dstrain)

        tang = self.ctang.copy()

        # alpha[0] stores alpha11-alpha22 (not the literal 3D alpha11).
        # alpha[4:6] store the physical shear backstresses.
        eta_trial = sigma[self.active] - alpha[self.active]
        smises = sqrt(dot(self.weights * eta_trial, eta_trial))

        deqpl = 0.0

        if smises > (1.0 + self.tolerance) * self.syield:

            # Reduced associative flow and kinematic hardening:
            #   dep = dp * W*eta/seq
            #   dalpha = dp * hard*B*eta/seq
            #   eta_trial = (I + t*A)*eta, t = dp/seq
            # with A = diag(C)*diag(W) + hard*diag(B).
            Cdiag = array([self.E, self.eg, self.eg])
            A = Cdiag * self.weights + self.hard * self.hard_weights

            def equivalent_stress(t):
                eta = eta_trial / (1.0 + t * A)
                return sqrt(dot(self.weights * eta, eta))

            # Bracket and solve the scalar yield condition.
            lo = 0.0
            hi = 1.0 / max(A)
            while equivalent_stress(hi) > self.syield:
                hi *= 2.0

            for _ in range(65):
                tmid = 0.5 * (lo + hi)
                if equivalent_stress(tmid) > self.syield:
                    lo = tmid
                else:
                    hi = tmid

            t = 0.5 * (lo + hi)
            eta = eta_trial / (1.0 + t * A)
            deqpl = t * self.syield
            flow = eta / self.syield

            # Update backstress and plastic/elastic strains.
            alpha[self.active] += self.hard * self.hard_weights * flow * deqpl
            deps_pl = self.weights * flow * deqpl
            eplas[self.active] += deps_pl
            eelas[self.active] -= deps_pl

            # Final stress: sigma = sigma_trial - C*deps_pl.
            sigma[self.active] -= Cdiag * deps_pl

            # Consistent algorithmic tangent from the scalar return mapping.
            d = 1.0 / (1.0 + t * A)
            v = self.weights * eta * A * d
            denom = dot(v, eta)
            dt_dtrial = self.weights * eta * d / denom
            deta_dtrial = diag(d) - outer(eta * A * d, dt_dtrial)

            # d sigma / d sigma_trial, then chain with elastic predictor.
            J = eye(3) - diag(Cdiag * self.weights) @ (
                t * deta_dtrial + outer(eta, dt_dtrial)
            )
            tang_reduced = J @ diag(Cdiag)

            tang = zeros(shape=(6, 6))
            tang[ix_(self.active, self.active)] = tang_reduced

        # Ensure inactive stresses remain identically zero.
        sigma[1:4] = 0.0

        self.setHistoryParameter(elementID, intpointID, "eelas", eelas)
        self.setHistoryParameter(elementID, intpointID, "eplas", eplas)
        self.setHistoryParameter(elementID, intpointID, "alpha", alpha)
        self.setHistoryParameter(elementID, intpointID, "sigma", sigma)

        return sigma, tang
