import numpy as np
from cardillo.utility.check_time_derivatives import check_time_derivatives


class ActuatedConstraint:
    def __init__(self, subsystem, tau, active=True, inactive_force=np.zeros(1)):
        self.subsystem = subsystem
        self.update_actuation(tau=tau, active=active, inactive_force=inactive_force)

        # TODO: do we have to update these as well?
        self.nin = 1
        self.W_in = self.W_g

        # TODO: mark this as an input somehow to use in FRF
        # TODO: allow for velocity constraint as well

    def update_actuation(self, *, tau=None, active=None, inactive_force=None):
        if tau is not None:
            self.tau, self.tau_dot, self.tau_ddot = check_time_derivatives(
                tau, None, None
            )

        if inactive_force is not None:
            if callable(inactive_force):
                self.inactive_force = inactive_force
            else:
                self.inactive_force = lambda t: inactive_force

            assert self.inactive_force(0).shape == (1,)

        if active is not None:
            self.active = active

        if self.active:
            assert hasattr(self, "tau")
            self.nla_g = 1
            self.g = self._g

            if hasattr(self, "h"):
                del self.h
                del self.h_q

        else:
            assert hasattr(self, "inactive_force")
            self.h = self._h
            self.h_q = self._h_q

            if hasattr(self, "nla_g"):
                del self.nla_g
                del self.g

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF
        self._nq = len(self.qDOF)
        self.uDOF = self.subsystem.uDOF
        self._nu = len(self.uDOF)

    ############
    # inactive #
    ############
    def _h(self, t, q, u):
        return self.subsystem.W_l(t, q) * self.inactive_force(t)

    def _h_q(self, t, q, u):
        return self.subsystem.W_l_q(t, q) * self.inactive_force(t)

    def KN_h(self, t, q, u):
        return self.subsystem.KN_l(t, q, self.inactive_force(t))

    ##########
    # active #
    ##########
    def _g(self, t, q):
        return self.subsystem.l(t, q) - self.tau(t)

    def g_q(self, t, q):
        return self.subsystem.l_q(t, q)

    def g_dot(self, t, q, u):
        return self.subsystem.l_dot(t, q, u) - self.tau_dot(t)

    def g_dot_q(self, t, q, u):
        return self.subsystem.l_dot_q(t, q, u)

    def g_dot_u(self, t, q):
        return self.subsystem.l_dot_u(t, q)

    def g_ddot(self, t, q, u, u_dot):
        return self.subsystem.l_ddot(t, q, u, u_dot) - self.tau_ddot(t)

    def W_g(self, t, q):
        return self.subsystem.W_l(t, q)

    def Wla_g_q(self, t, q, la_g):
        return self.subsystem.Wla_l_q(t, q, la_g)
        # return np.einsum("ijk,j->ik", self.subsystem.W_l_q(t, q), la_g)

    def KN_g(self, t, q, la_g):
        print(la_g)
        return self.subsystem.KN_l(t, q, la_g)
