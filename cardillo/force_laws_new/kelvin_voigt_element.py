import numpy as np

from ._base import ScalarForceLaw


class KelvinVoigtElement(ScalarForceLaw):
    r"""Scalar force law consisting of a linear spring and a linear damper acting in parallel.

    Parameters
    ----------
    subsystem : object
        Object providing the interface for a scalar force law, e.g., Revolute, TwoPointInteraction
    k : float
        Stiffness of spring.
    d : float
        Viscosity of linear damper.
    l_ref : float
        Undeformed elongation of linear spring.
    compliance_form : bool
        True : Scalar force is contemplated as Lagrange multiplier and force law as algebraic equation resulting in a DAE.
        False : Scalar force is contemplated as function of the system's state resulting in an ODE.
    name : str
        Name of contribution.
    """

    def __init__(
        self,
        subsystem,
        k,
        d,
        l_ref=None,
        compliance_form=True,
        name="kelvin_voigt_element",
    ):
        nla_d = 1 if d > 0.0 else 0
        super().__init__(subsystem, nla_d, compliance_form)
        assert k > 0.0
        self.__k = k
        self.__d = d
        self.l_ref = l_ref
        self.name = name

        # e_la_e = 1 / k and d_la_d = 1 / d are constant
        self.constant_e_la_e = True
        self.constant_d_la_d = True

    def assembler_callback(self):
        super().assembler_callback()
        if self.l_ref is None:
            self.l_ref = self.subsystem.l(self.subsystem.t0, self.subsystem.q0)

    # stiffness
    def _E_pot(self, t, l):
        return 0.5 * self.__k * (l - self.l_ref) ** 2

    def _E_pot_comp_e(self, t, l, la_e):
        return -(l - self.l_ref) * la_e - 0.5 / self.__k * la_e**2

    def _la_e(self, t, l):
        return -self.__k * (l - self.l_ref)

    def _la_e_l(self, t, l):
        return -self.__k

    def _e(self, t, l, la_e):
        return la_e / self.__k + (l - self.l_ref)

    def _e_l(self, t, l, la_e):
        return 1.0

    def _e_la_e(self, t, l):
        return 1 / self.__k

    # damping
    def _la_d(self, t, l, l_dot):
        return -self.__d * l_dot

    def _la_d_l(self, t, l, l_dot):
        return 0.0

    def _la_d_l_dot(self, t, l):
        return -self.__d

    def _d(self, t, l, l_dot, la_d):
        return la_d / self.__d + l_dot

    def _d_l(self, t, l, l_dot, la_d):
        return 0.0

    def _d_l_dot(self, t, l):
        return 1.0

    def _d_la_d(self, t, l):
        return 1 / self.__d
