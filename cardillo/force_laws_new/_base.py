from abc import ABC, abstractmethod
import numpy as np


class ScalarForceLaw(ABC):
    r"""Base class for scalar force laws acting along a scalar generalized
    elongation l(t, q) of a subsystem, split into an elastic and a damping part.

    The total scalar force la = la_e + la_d enters the kinetic equation as
    W_l(t, q) * la, i.e., la > 0 tends to increase l. Both forces are
    contemplated as Lagrange multipliers determined by implicit force laws:

    - elasticity (nla_e = 1):

      e(t, l, la_e) = e_la_e(t, l) * la_e + e(t, l, 0) = 0,

      i.e., e is affine in la_e.

    - damping (nla_d = 0 or 1):

      d(t, l, l_dot, la_d) = d_la_d(t, l) * la_d + d(t, l, l_dot, 0) = 0,

      i.e., d is affine in la_d and, since l_dot = W_l(t, q).T @ u, linear in u.

    If e_la_e or d_la_d do not depend on l, the derived class should set
    constant_e_la_e = True or constant_d_la_d = True, respectively, such that
    the system evaluates them only once during assembly.

    Derived classes implement the scalar functions in terms of (t, l, l_dot,
    la_e, la_d), i.e., _la_e, _e, _e_la_e, _E_pot, ... for the elastic part and
    _la_d, _d, _d_la_d, ... for the damping part. The base class maps them via
    the chain rule to functions of (t, q, u).

    Parameters
    ----------
    subsystem : object
        Object providing the interface for a scalar force law, e.g.,
        TwoPointInteraction, Revolute or Prismatic. It has to provide l, l_q,
        l_dot, l_dot_q, l_dot_u, W_l, W_l_q, W2_l and the local numbers of
        coordinates _nq and _nu.
    nla_d : int
        Number of damping multipliers; 0 (no damping) or 1.
    compliance_form : bool
        True : Scalar forces are contemplated as Lagrange multipliers la_e and la_d
        and the force laws as algebraic equations e = 0 and d = 0.
        False : Scalar force is contemplated as function of the system's state
        (not implemented yet).
    """

    def __init__(self, subsystem, nla_d, compliance_form=True):
        self.subsystem = subsystem
        self.l = self.subsystem.l
        self.l_q = self.subsystem.l_q
        self.l_dot = self.subsystem.l_dot
        self.l_dot_q = self.subsystem.l_dot_q
        self.l_dot_u = self.subsystem.l_dot_u

        if compliance_form:
            # elasticity
            self.nla_e = 1

            # energy
            self.E_pot_comp_e = lambda t, q, la_e: self._E_pot_comp_e(
                t, self.l(t, q), la_e
            )

            # kinetic equation
            self.W_e = self._W
            self.Wla_e_q = lambda t, q, la_e: self._Wla_q(t, q, la_e)
            self.KN_e = lambda t, q, la_e: self._KN(t, q, la_e)

            # elasticity equation
            # e(t, l, la_e) = e_la_e(t, l) @ la_e + e(t, l, 0)
            self.e = lambda t, q, la_e: self._e(t, self.l(t, q), la_e)
            self.e_q = lambda t, q, la_e: self._e_l(
                t, self.l(t, q), la_e
            ) * self.subsystem.l_q(t, q)
            self.e_la_e = lambda t, q: self._e_la_e(t, self.l(t, q))

            if nla_d > 0:
                # damping
                self.nla_d = nla_d

                # kinetic equation
                self.W_d = self._W
                self.Wla_d_q = lambda t, q, la_d: self._Wla_q(t, q, la_d)
                self.DG_d = self._DG_d
                self.KN_d = lambda t, q, la_d: self._KN(t, q, la_d)

                # damping equation
                # d(t, l, l_dot, la_d) = d_la_d(t, l) @ la_d + d(t, l, l_dot, 0)
                # d(t, l(t, q), l_dot(t, q, u), 0) = W(t, q).T @ u + chi(t, q)
                self.d = lambda t, q, u, la_d: self._d(
                    t, self.l(t, q), self.l_dot(t, q, u), la_d
                )
                self.d_q = lambda t, q, u, la_d: (
                    self._d_l(t, self.l(t, q), self.l_dot(t, q, u), la_d)
                    * self.subsystem.l_q(t, q)
                    + self._d_l_dot(t, self.l(t, q)) * self.subsystem.l_dot_q(t, q, u)
                )
                self.d_la_d = lambda t, q: self._d_la_d(t, self.l(t, q))

        else:
            self.E_pot = lambda t, q: self._E_pot(t, self.l(t, q))
            self.h = self._h
            self.h_q = self._h_q
            self.h_u = self._h_u
            self.KN_h = self._KN_h
            self.DG_h = self._DG_h

        # solved elastic and daming equation
        self.la_e = lambda t, q: self._la_e(t, self.l(t, q))
        self.la_e_q = lambda t, q: self._la_e_l(t, self.l(t, q)) * self.l_q(t, q)
        self.la_d = lambda t, q, u: self._la_d(t, self.l(t, q), self.l_dot(t, q, u))
        self.la_d_q = lambda t, q, u: (
            self._la_d_l(t, self.l(t, q), self.l_dot(t, q, u))
            * self.subsystem.l_q(t, q)
            + self._la_d_l_dot(t, self.l(t, q)) * self.subsystem.l_dot_q(t, q, u)
        )
        self.la_d_u = lambda t, q: self._la_d_l_dot(t, self.l(t, q)) * self.l_dot_u(
            t, q
        )

    def assembler_callback(self):
        self.subsystem.assembler_callback()
        self.qDOF = self.subsystem.qDOF
        self.uDOF = self.subsystem.uDOF

    # elasticity
    @abstractmethod
    def _E_pot(self, t, l): ...

    @abstractmethod
    def _E_pot_comp_e(self, t, l, la_e): ...

    @abstractmethod
    def _la_e(self, t, l): ...

    @abstractmethod
    def _la_e_l(self, t, l): ...

    @abstractmethod
    def _e(self, t, l, la_e): ...

    @abstractmethod
    def _e_l(self, t, l, la_e): ...

    @abstractmethod
    def _e_la_e(self, t, l): ...

    # damping
    @abstractmethod
    def _la_d(self, t, l, l_dot): ...

    @abstractmethod
    def _la_d_l(self, t, l, l_dot): ...

    @abstractmethod
    def _la_d_l_dot(self, t, l): ...

    @abstractmethod
    def _d(self, t, l, l_dot, la_d): ...

    @abstractmethod
    def _d_l(self, t, l, l_dot, la_d): ...

    @abstractmethod
    def _d_l_dot(self, t, l): ...

    @abstractmethod
    def _d_la_d(self, t, l): ...

    # common functions
    def _DG_d(self, t, q, la_d):
        raise NotImplementedError("DG_d is not implemented for scalar force laws.")

    def _W(self, t, q):
        return self.subsystem.W_l(t, q).reshape(self.subsystem._nu, self.nla_e)

    def _Wla_q(self, t, q, la):
        return la * self.subsystem.W_l_q(t, q).reshape(
            self.subsystem._nu, self.subsystem._nq
        )

    def _KN(self, t, q, la):
        W2_l = self.subsystem.W2_l(t, q).reshape(self.subsystem._nu, self.subsystem._nu)

        # TODO: test this and check if there is no N?
        # TODO: check all the signs below here!
        K = -la * W2_l
        return K, np.zeros((self.subsystem._nu, self.subsystem._nu))

    def _h(self, t, q, u):
        la = self.la_e(t, q) + self.la_d(t, q, u)
        return la * self.subsystem.W_l(t, q).reshape(self.subsystem._nu)

    def _h_q(self, t, q, u):
        la = self.la_e(t, q) + self.la_d(t, q, u)
        la_q = self.la_e_q(t, q) + self.la_d_q(t, q, u)
        return la * self.subsystem.W_l_q(t, q).reshape(
            self.subsystem._nu, self.subsystem._nq
        ) + np.outer(self.subsystem.W_l(t, q).reshape(self.subsystem._nu), la_q)

    def _h_u(self, t, q, u):
        la_u = self.la_d_u(t, q)
        return np.outer(self.subsystem.W_l(t, q).reshape(self.subsystem._nu), la_u)

    def _KN_h(self, t, q, u):
        raise NotImplementedError
        l = self.l(t, q)
        l_dot = self.l_dot(t, q, u)

        la_c = self._la_c(t, l, l_dot)
        la_c_l = self._la_c_l(t, l, l_dot)
        la_c_l_dot = self._la_c_l_dot(t, l, l_dot)

        W_l = self.subsystem.W_l(t, q).reshape(self.subsystem._nu)
        W2_l = self.subsystem.W2_l(t, q).reshape(self.subsystem._nu, self.subsystem._nu)

        # TODO: test this and check if N is purely skew-symmetric!
        # TODO: check all the signs below here!
        K = -la_c * W2_l - np.outer(W_l, la_c_l * W_l)
        N = -np.outer(W_l, W2_l @ u) * la_c_l_dot
        return K, N

    def _DG_h(self, t, q, u):
        raise NotImplementedError
        # TODO: this should be equal to +- self._h_u(t, q, u)
        la_c_l_dot = self._la_c_l_dot(t, self.l(t, q), self.l_dot(t, q, u))
        W_l = self.subsystem.W_l(t, q).reshape(self.subsystem._nu)
        D = -np.outer(W_l, la_c_l_dot * W_l)
        return D, np.zeros((self.subsystem._nu, self.subsystem._nu))

    def export(self, sol_i, **kwargs):
        return self.subsystem.export(sol_i, **kwargs)

    def export_blender(self, *args, **kwargs):
        return self.subsystem.export_blender(*args, **kwargs)

    def export_blender_modes(self, *args, **kwargs):
        return self.subsystem.export_blender_modes(*args, **kwargs)
