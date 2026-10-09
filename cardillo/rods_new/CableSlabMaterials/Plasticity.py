# SPDX-License-Identifier: MIT
# Copyright (c) 2011–2026 Joris J.C. Remmers

from .BaseMaterial import BaseMaterial
from .MatUtils import vonMisesStress, hydrostaticStress
from .MatUtils import transform3To2, transform2To3
from numpy import zeros, ones, dot, array, outer
from math import sqrt


class Plasticity(BaseMaterial):
    """
    Rate-independent plasticity model with linear kinematic hardening.

    This class implements an elastoplastic constitutive model based on the
    von Mises yield criterion. The material behaves as isotropic linear
    elastic below the yield stress. Once yielding occurs, plastic deformation
    is computed using an associative flow rule and the yield surface evolves
    according to a linear kinematic hardening law.

    Kinematic hardening is represented by a backstress tensor, allowing the
    yield surface to translate in stress space during plastic deformation.
    This enables the model to describe different yield stresses during load
    reversal (Bauschinger effect).

    Required parameters
    -------------------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    syield : float
        Initial von Mises yield stress.
    hard : float
        Linear kinematic hardening modulus.
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

        self.ebulk3 = self.E / (1.0 - 2.0 * self.nu)
        self.eg2 = self.E / (1.0 + self.nu)
        self.eg = 0.5 * self.eg2
        self.eg3 = 3.0 * self.eg
        self.elam = (self.ebulk3 - self.eg2) / 3.0

        self.ctang = zeros(shape=(6, 6))

        self.ctang[:3, :3] = self.elam

        self.ctang[0, 0] += self.eg2
        self.ctang[1, 1] = self.ctang[0, 0]
        self.ctang[2, 2] = self.ctang[0, 0]

        self.ctang[3, 3] = self.eg
        self.ctang[4, 4] = self.ctang[3, 3]
        self.ctang[5, 5] = self.ctang[3, 3]

        # ------------------------------------------------------------------------------
        #  pre:  kinematics object containing current strain (kinemtics.strain)
        #  post: stress vector and tangent matrix
        # ------------------------------------------------------------------------------

    def getStress(self, kinematics, actions, element, intpointID):

        elementID = element.id

        self.initializeMaterialPoint(elementID, intpointID)

        eelas = self.getHistoryParameter(elementID, intpointID, "eelas")
        eplas = self.getHistoryParameter(elementID, intpointID, "eplas")
        alpha = self.getHistoryParameter(elementID, intpointID, "alpha")
        sigma = self.getHistoryParameter(elementID, intpointID, "sigma")

        if len(kinematics.dstrain) == 6:
            dstrain = kinematics.dstrain
        else:
            dstrain = transform2To3(kinematics.dstrain)

        eelas += dstrain

        sigma += dot(self.ctang, dstrain)

        tang = self.ctang

        smises = vonMisesStress(sigma - alpha)

        deqpl = 0.0

        if smises > (1.0 + self.tolerance) * self.syield:
            shydro = hydrostaticStress(sigma)

            flow = sigma - alpha

            flow[:3] = flow[:3] - shydro * ones(3)
            flow *= 1.0 / smises

            deqpl = (smises - self.syield) / (self.eg3 + self.hard)

            alpha += self.hard * flow * deqpl
            eplas[:3] += 1.5 * flow[:3] * deqpl
            eelas[:3] += -1.5 * flow[:3] * deqpl

            eplas[3:] += 3.0 * flow[3:] * deqpl
            eelas[3:] += -3.0 * flow[3:] * deqpl

            sigma = alpha + flow * self.syield
            sigma[:3] += shydro * ones(3)

            effg = self.eg * (self.syield + self.hard * deqpl) / smises
            effg2 = 2.0 * effg
            effg3 = 3.0 * effg
            efflam = 1.0 / 3.0 * (self.ebulk3 - effg2)
            effhdr = self.eg3 * self.hard / (self.eg3 + self.hard) - effg3

            tang = zeros(shape=(6, 6))
            tang[:3, :3] = efflam

            for i in range(3):
                tang[i, i] += effg2
                tang[i + 3, i + 3] += effg

            tang += effhdr * outer(flow, flow)

        self.setHistoryParameter(elementID, intpointID, "eelas", eelas)
        self.setHistoryParameter(elementID, intpointID, "eplas", eplas)
        self.setHistoryParameter(elementID, intpointID, "alpha", alpha)
        self.setHistoryParameter(elementID, intpointID, "sigma", sigma)

        if len(kinematics.dstrain) == 6:
            return sigma, tang
        else:
            return transform3To2(sigma, tang)
