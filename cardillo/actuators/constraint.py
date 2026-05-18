import numpy as np
from cardillo.utility.check_time_derivatives import check_time_derivatives


class ActuatedConstraint:
    def __init__(self, subsystem, tau):
        self.subsystem = subsystem
        self.update_actuation(tau)
        self.nla_g = 1

        # TODO: mark this as an input somehow to use in FRF
        # TODO: allow for velocity constraint as well

    def update_actuation(self, tau):
        self.tau, self.tau_dot, self.tau_ddot = check_time_derivatives(tau, None, None)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF
        self._nq = len(self.qDOF)
        self.uDOF = self.subsystem.uDOF
        self._nu = len(self.uDOF)

    def g(self, t, q):
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
