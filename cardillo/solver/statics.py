import numpy as np
import warnings
from scipy.sparse import lil_array, bmat, csc_array
from scipy.sparse.linalg import inv as sparse_inv
from scipy.linalg import eigh, null_space
from tqdm import tqdm

from cardillo.math.fsolve import fsolve
from cardillo.solver import Solution, SolverOptions, SolverSummary
from cardillo.utility.coo_matrix import CooMatrix


class Newton:
    """Force and displacement controlled Newton-Raphson method. This solver
    is used to find a static solution for a mechanical system. Forces and
    bilateral constraint functions are incremented in each load step if they
    depend on the time t in [0, 1]. Thus, a force controlled Newton-Raphson method
    is obtained by constructing a time constant constraint function function.
    On the other hand a displacement controlled Newton-Raphson method is
    obtained by passing constant forces and time dependent constraint functions.
    """

    def __init__(
        self,
        system,
        n_load_steps=1,
        verbose=True,
        updated=False,
        options=SolverOptions(),
    ):
        self.system = system
        self.options = options
        self.verbose = verbose
        self.updated = updated
        self.load_steps = np.linspace(0, 1, n_load_steps + 1)
        self.nt = len(self.load_steps)

        self.len_t = len(str(self.nt))
        self.len_maxIter = len(str(self.options.newton_max_iter))

        # other dimensions
        self.nq = system.nq
        self.nu = system.nu
        self.nla_N = system.nla_N

        self.split_f = np.cumsum(
            np.array(
                [system.nu, system.nla_g, system.nla_c, system.nla_N],
                dtype=int,
            )
        )
        self.split_x = np.cumsum(
            np.array(
                [system.nq, system.nla_g, system.nla_c],
                dtype=int,
            )
        )

        if self.updated:
            self.nx_bar = system.nu + system.nla_g + system.nla_c + system.nla_N
            # not sure how the Jacobian is for this
            assert self.nla_N == 0

            def update_rule(x, Delta_x_bar, t):
                q, la_g, la_c, la_N = np.array_split(x, self.split_x)
                ds, dla_g, dla_c, dla_N, _ = np.array_split(Delta_x_bar, self.split_f)
                dq = self.system.q_dot(t, q, ds)
                dx = np.zeros_like(x)
                dx[: self.split_x[0]] = dq
                dx[self.split_x[0] : self.split_x[1]] = dla_g
                dx[self.split_x[1] : self.split_x[2]] = dla_c
                dx[self.split_x[2] :] = dla_N
                return dx

            self.update_rule = update_rule

        else:
            self.nx_bar = system.nq + system.nla_g + system.nla_c + system.nla_N
            self.update_rule = None

        # initial conditions
        x0 = np.concatenate((system.q0, system.la_g0, system.la_c0, system.la_N0))
        nx = len(x0)
        self.u0 = np.zeros(system.nu)  # zero velocities as system is static

        print(f"{self.nx_bar = }, {nx = }")

        # pre-evaluate compliance matrix
        self.c_la_c = self.system.c_la_c()

        # memory allocation
        self.x = np.zeros((self.nt, nx), dtype=float)
        self.x[0] = x0

        self.all_x = np.zeros([len(self.x[:, 0])], dtype=object)
        self.all_x[0] = np.array([self.x[0]])

        # step callback
        def update_callback(x, t):
            x[: self.split_x[0]], _ = self.system.step_callback(
                t, x[: self.split_x[0]], self.u0
            )
            return x

        self.update_callback = update_callback

    def fun(self, x, t):
        # unpack unknowns
        q, la_g, la_c, la_N = np.array_split(x, self.split_x)

        # evaluate quantites that are required for computing the residual and
        # the jacobian
        # csr is used for efficient matrix vector multiplication, see
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array
        self.W_g = self.system.W_g(t, q, format="csr")
        self.W_c = self.system.W_c(t, q, format="csr")
        self.W_N = self.system.W_N(t, q, format="csr")
        self.g_N = self.system.g_N(t, q)

        # static equilibrium
        F = np.zeros(self.nx_bar)
        F[: self.split_f[0]] = (
            self.system.h(t, q, self.u0)
            + self.W_g @ la_g
            + self.W_c @ la_c
            + self.W_N @ la_N
        )
        F[self.split_f[0] : self.split_f[1]] = self.system.g(t, q)
        F[self.split_f[1] : self.split_f[2]] = self.system.c(t, q, self.u0, la_c)
        F[self.split_f[2] : self.split_f[3]] = np.minimum(la_N, self.g_N)
        if not self.updated:
            F[self.split_f[3] :] = self.system.g_S(t, q)
        return F

    def jac_updated(self, x, t):
        # unpack unknowns
        q, la_g, la_c, la_N = np.array_split(x, self.split_x)

        # evaluate additionally required quantites for computing the jacobian
        # coo is used for efficient bmat
        KNs = [
            self.system.KN_h(t, q, self.u0),
            self.system.KN_g(t, q, la_g),
            self.system.KN_c(t, q, la_c),
            self.system.KN_N(t, q, la_N),
        ]
        K = np.sum([KN[0] for KN in KNs])
        N = np.sum([KN[1] for KN in KNs])

        # note: csr_matrix is best for row slicing, see
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array
        dRla_N = lil_array((self.nla_N, self.nu), dtype=float)
        Rla_N_la_N = lil_array((self.nla_N, self.nla_N), dtype=float)
        for i in range(self.nla_N):
            if la_N[i] < self.g_N[i]:
                Rla_N_la_N[i, i] = 1.0
            else:
                dRla_N[i] = self.W_N.T[i]

        # fmt: off
        return bmat([[     K + N, self.W_g,    self.W_c,   self.W_N], 
                     [self.W_g.T,     None,        None,       None],
                     [self.W_c.T,     None, self.c_la_c,       None],
                     [    dRla_N,     None,        None, Rla_N_la_N],], format="csc")
        # fmt: on

    def jac(self, x, t):
        # unpack unknowns
        q, la_g, la_c, la_N = np.array_split(x, self.split_x)

        # evaluate additionally required quantites for computing the jacobian
        # coo is used for efficient bmat
        K = (
            self.system.h_q(t, q, self.u0)
            + self.system.Wla_g_q(t, q, la_g)
            + self.system.Wla_c_q(t, q, la_c)
            + self.system.Wla_N_q(t, q, la_N)
        )
        g_q = self.system.g_q(t, q)
        g_S_q = self.system.g_S_q(t, q)
        c_q = self.system.c_q(t, q, self.u0, la_c)

        # note: csr_matrix is best for row slicing, see
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array
        g_N_q = self.system.g_N_q(t, q, format="csr")

        Rla_N_q = lil_array((self.nla_N, self.nq), dtype=float)
        Rla_N_la_N = lil_array((self.nla_N, self.nla_N), dtype=float)
        for i in range(self.nla_N):
            if la_N[i] < self.g_N[i]:
                Rla_N_la_N[i, i] = 1.0
            else:
                Rla_N_q[i] = g_N_q[i]

        # fmt: off
        return bmat([[      K, self.W_g,    self.W_c,   self.W_N], 
                     [    g_q,     None,        None,       None],
                     [    c_q,     None, self.c_la_c,       None],
                     [Rla_N_q,     None,        None, Rla_N_la_N],
                     [  g_S_q,     None,        None,       None],], format="csc")
        # fmt: on

    def __pbar_text(self, force_iter, newton_iter, error):
        return (
            f" force iter {force_iter+1:>{self.len_t}d}/{self.nt};"
            f" Newton steps {newton_iter+1:>{self.len_maxIter}d}/{self.options.newton_max_iter};"
            f" error {error:.4e}"
        )

    def solve(self):
        self.solver_summary = SolverSummary(
            f"Newton{' updated' if self.updated else ''}"
        )
        pbar = range(0, self.nt)
        if self.verbose:
            pbar = tqdm(pbar, leave=True)
        for i in pbar:
            sol = fsolve(
                self.fun,
                self.x[i],
                jac=self.jac if not self.updated else self.jac_updated,
                fun_args=(self.load_steps[i],),
                jac_args=(self.load_steps[i],),
                update_rule=self.update_rule,
                # update_callback=self.update_callback if self.updated else None,
                update_args=(self.load_steps[i],),
                options=self.options,
            )
            self.x[i] = sol.x
            self.all_x[i] = sol.all_x
            if self.verbose:
                pbar.set_description(self.__pbar_text(i, sol.nit, sol.error))
            self.solver_summary.add_newton(sol.nit, sol.error, sol.final_quadratic_rate)

            if not sol.success and not self.options.continue_with_unconverged:
                # return solution up to this iteration
                if self.verbose:
                    pbar.close()
                print(
                    f"Newton-Raphson method not converged, returning solution "
                    f"up to iteration {i+1:>{self.len_t}d}/{self.nt}"
                )

                # put iterates into solution
                sub_ts = np.linspace(0, 1, self.options.newton_max_iter + 5)
                all_q = np.vstack(
                    [xi[:, : self.split_x[0]] for xi in self.all_x[: i + 1]]
                )
                all_t = np.concatenate(
                    [j + sub_ts[: len(self.all_x[j][:, 0])] for j in range(i + 1)]
                )
                return Solution(
                    system=self.system,
                    t=self.load_steps[: i + 1],
                    q=self.x[: i + 1, : self.split_x[0]],
                    u=np.zeros((i + 1, self.nu)),
                    la_g=self.x[: i + 1, self.split_x[0] : self.split_x[1]],
                    la_c=self.x[: i + 1, self.split_x[1] : self.split_x[2]],
                    la_N=self.x[: i + 1, self.split_x[2] :],
                    all_x=self.all_x[: i + 1],
                    all_q=all_q,
                    all_t=all_t,
                    solver_summary=self.solver_summary,
                )

            # # solver step callback
            # self.x[i] = self.update_callback(self.x[i], self.load_steps[i])

            self.x[i, : self.split_x[0]], _ = self.system.step_callback(
                self.load_steps[i], self.x[i, : self.split_x[0]], self.u0
            )

            # warm start for next step; store solution as new initial guess
            if i < self.nt - 1:
                self.x[i + 1] = self.x[i]

        # return solution object
        if self.verbose:
            pbar.close()

        # put iterates into solution
        sub_ts = np.linspace(0, 1, self.options.newton_max_iter + 5)
        all_q = np.vstack([xi[:, : self.split_x[0]] for xi in self.all_x[: i + 1]])
        all_t = np.concatenate(
            [j + sub_ts[: len(self.all_x[j][:, 0])] for j in range(i + 1)]
        )
        return Solution(
            self.system,
            t=self.load_steps,
            q=self.x[: i + 1, : self.split_x[0]],
            u=np.zeros((len(self.load_steps), self.nu)),
            la_g=self.x[: i + 1, self.split_x[0] : self.split_x[1]],
            la_c=self.x[: i + 1, self.split_x[1] : self.split_x[2]],
            la_N=self.x[: i + 1, self.split_x[2] :],
            all_x=self.all_x[: i + 1],
            all_q=all_q,
            all_t=all_t,
            solver_summary=self.solver_summary,
        )


# read https://doi.org/10.1016/j.engstruct.2020.111755
class Riks:
    """Linear arc-length solver close to Riks method as dervied in Crisfield1991 
    section 9.3.2 p.273. A variable arc-length is chosen as shown by 
    Crisfield1981 or Crisfield 1983. For the first predictor a tangent predictor 
    is used. For all other predictors a simple secant predictor is sufficient. 
    This enables the solver to 'run forward' instead of 'doubling back on its track'.

    References
    ----------
    - stackexchange : https://scicomp.stackexchange.com/a/28140 \\
    - Wempner1971: https://doi.org/10.1016/0020-7683(71)90038-2 \\
    - Riks1972: https://doi.org/10.1115/1.3422829 \\
    - Riks1979: https://doi.org/10.1016/0020-7683(79)90081-7 \\
    - Crsfield1981: https://doi.org/10.1016/0045-7949(81)90108-5 \\
    - Crisfield1991: http://freeit.free.fr/Finite%20Element/Crisfield%20M.A.%20Vol.1.%20Non-Linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Essentials%20(Wiley,19.pdf \\
    - Crisfield1996: http://inis.jinr.ru/sl/M_Mathematics/MN_Numerical%20methods/MNf_Finite%20elements/Crisfield%20M.A.%20Vol.2.%20Non-linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Advanced%20Topics%20(Wiley,1996)(ISBN%20047195649X)(509s).pdf \\
    - Neto1999: https://doi.org/10.1016/S0045-7825(99)00042-0
    """

    def __init__(
        self,
        system,
        iter_goal=4,
        la_arc0=1.0e-3,
        la_arc_span=np.array([0, 1], dtype=float),
        scale_exponent=0.5,
        max_load_steps=int(1e4),
        options=SolverOptions(),
    ):
        self.system = system
        self.options = options
        self.la_arc0 = la_arc0
        self.la_arc_span = la_arc_span
        self.max_load_steps = max_load_steps

        # initial arc-length parameter is not required in the first step and
        # will be computed later
        self.ds = 0

        # step size of finite differences
        self.eps = self.options.numerical_jacobian_eps

        # parameter for the step size scaling
        self.iter_goal = iter_goal
        self.MIN_FACTOR = 0.25  # minimal scaling factor
        self.MAX_FACTOR = 1.5  # maximal scaling factor
        self.scale_exponent = scale_exponent

        # split vectors
        self.split_unknowns = np.cumsum(
            np.array(
                [
                    system.nq,
                    system.nla_c,
                    system.nla_g,
                    system.nla_N,
                    1,
                ],
                dtype=int,
            )
        )[:-1]
        self.split_residual = np.cumsum(
            np.array(
                [
                    system.nu,
                    system.nla_c,
                    system.nla_g,
                    system.nla_S,
                    system.nla_N,
                    1,
                ],
                dtype=int,
            )
        )[:-1]

        # initial
        self.q0 = self.system.q0
        self.la_c0 = self.system.la_c0
        self.la_g0 = self.system.la_g0
        self.la_arc0 = la_arc0
        self.la_N0 = self.system.la_N0
        self.u0 = np.zeros(system.nu)  # statics

        # initial values for generalized coordinates, lagrange multipliers and force scaling
        self.xk = np.concatenate(
            (self.q0, self.la_c0, self.la_g0, self.la_N0, np.array([0]))
        )
        self.x0_bar = np.concatenate(
            (self.q0, self.la_c0, self.la_g0, self.la_N0, np.array([la_arc0]))
        )

        ####################################################################################################
        # Solve linearized system for fixed external force using Newtons method.
        # From this solution we can extract the initial ds using the arc length equation.
        # All other ds values will be modified according to the number of used Newton steps,
        # see https://scicomp.stackexchange.com/questions/28137/initialize-arc-length-control-in-riks-method
        ####################################################################################################
        print(f"solve equilibrium for given initial la_arc0")

        def fun(x):
            x = np.concatenate((x, [la_arc0]))
            return self.R(x)[:-1]

        def jac(x):
            x = np.concatenate((x, [la_arc0]))
            return self.J(x)[:-1, :-1]

        sol = fsolve(fun, self.x0_bar[:-1], jac=jac, options=options)
        assert (
            sol.success
        ), "solving for initial arc-length parameter 'ds' did not converge => chose another 'la_arc0'"

        # compute initial ds from arc-length equation
        self.x0_bar = np.concatenate((sol.x, [la_arc0]))
        self.ds = self.a(self.x0_bar) ** 0.5
        assert self.ds > 0, "initial ds is zero"
        print(f"initial ds: {self.ds:2.4e}")

    def a(self, x):
        """The most primitive arc-length equation restricts the change of all
        generalized coordinates `qn1` w.r.t. the last converged Newton step `qn`."""
        qn = np.array_split(self.xk, self.split_unknowns)[0]
        qn1 = np.array_split(x, self.split_unknowns)[0]
        dq = qn1 - qn
        return dq @ dq

    def a_q(self, x):
        qn = np.array_split(self.xk, self.split_unknowns)[0]
        qn1 = np.array_split(x, self.split_unknowns)[0]
        dq = qn1 - qn
        return 2 * dq

    def R(self, x):
        # extract generalized coordinates, Lagrange multipliers and arc-length parameter
        q, la_c, la_g, la_N, t = np.array_split(x, self.split_unknowns)
        t = t[0]

        # evaluate all functions with t = la_arc
        # - this requires the external force that should be scaled to be of the form
        #   h(t, q) = W(g) * t
        # - for displacement control, the bilateral constraints can be time-dependent
        #   g = g(t, q)

        # compute quantities required for Jacobian
        self.W_g = self.system.W_g(t, q, format="csr")
        self.W_c = self.system.W_c(t, q, format="csr")
        self.W_N = self.system.W_N(t, q, format="csr")
        self.g_N = self.system.g_N(t, q)
        self.h = self.system.h(t, q, self.u0)
        self.g = self.system.g(t, q)

        # build residual
        R = np.zeros_like(x)
        R = x.copy()
        R[: self.split_residual[0]] = self.h + self.W_c @ la_c + self.W_g @ la_g
        R[self.split_residual[0] : self.split_residual[1]] = self.system.c(
            t, q, self.u0, la_c
        )
        R[self.split_residual[1] : self.split_residual[2]] = self.g
        R[self.split_residual[2] : self.split_residual[3]] = self.system.g_S(t, q)
        R[self.split_residual[3] : self.split_residual[4]] = np.minimum(la_N, self.g_N)
        R[-1] = self.a(x) - self.ds**2

        return R

    def J(self, x):
        # extract generalized coordinates, Lagrange multipliers and arc-length parameter
        q, la_c, la_g, la_N, t = np.array_split(x, self.split_unknowns)
        t = t[0]

        # evaluate additionally required quantites for computing the jacobian
        # coo is used for efficient bmat
        K = (
            self.system.h_q(t, q, self.u0)
            + self.system.Wla_c_q(t, q, la_c)
            + self.system.Wla_g_q(t, q, la_g)
            + self.system.Wla_N_q(t, q, la_N)
        )
        c_q = self.system.c_q(t, q, self.u0, la_c)
        c_la_c = self.system.c_la_c()
        g_q = self.system.g_q(t, q)
        g_S_q = self.system.g_S_q(t, q)

        # note: csr_matrix is best for row slicing, see
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array
        g_N_q = self.system.g_N_q(t, q, format="csr")

        Rla_N_q = lil_array((self.system.nla_N, self.system.nq), dtype=float)
        Rla_N_la_N = lil_array((self.system.nla_N, self.system.nla_N), dtype=float)
        for i in range(self.system.nla_N):
            if la_N[i] < self.g_N[i]:
                Rla_N_la_N[i, i] = 1.0
            else:
                Rla_N_q[i] = g_N_q[i]

        # note: We use finite differences to compute the derivatives w.r.t.
        # to the arc-length parameter. Hence, we do not have to specify here
        # how the arc-length parameter enters the vector of generalized forces h.
        # For displacement based approaches, we simply add a corresponding
        # bilateral constraint g(t, q).
        eps = self.eps
        Wla_g_t = (self.system.W_g(t + eps, q) @ la_g - self.W_g @ la_g) / eps
        h_t = (self.system.h(t + eps, q, self.u0) - self.h) / eps
        Ru_t = h_t + Wla_g_t
        g_t = (self.system.g(t + eps, q) - self.g) / eps

        # derivative of the arc length equation
        a_q = self.a_q(x)

        # fmt: off
        return bmat([[      K, self.W_c, self.W_g,   self.W_N, Ru_t[:, None]], 
                     [    c_q,   c_la_c,     None,       None,          None],
                     [    g_q,     None,     None,       None,  g_t[:, None]],
                     [  g_S_q,     None,     None,       None,          None],
                     [Rla_N_q,     None,     None, Rla_N_la_N,          None],
                     [    a_q,     None,     None,       None,          None]], format="csc")
        # fmt: on

    def solve(self):
        # count number of force increments to get first increment with tangential predictor
        i = 0

        # initialize current generalized coordinates, Lagrange multipliers and
        # arc-length parameter
        q = [self.q0]
        la_c = [self.la_c0]
        la_g = [self.la_g0]
        la_N = [self.la_N0]
        la_arc = [self.la_arc0]

        # loop over ranges of force scaling
        xk1 = self.x0_bar.copy()  # initialize such that Jacobian is regular!

        # progress bar
        pbar = tqdm(total=100, leave=True)
        i0 = 0
        load_step = 0
        while (
            xk1[-1] >= self.la_arc_span[0]
            and xk1[-1] <= self.la_arc_span[1]
            and load_step <= self.max_load_steps
        ):
            # increment number of steps
            i += 1
            # load step counter
            load_step += 1

            # use secant predictor for all other force increments than the first one
            if i > 1:
                # secand predictor for all but the first newton iteration
                dx = self.xk - self.x0
                xk1 += dx

            # solve nonlinear system
            sol = fsolve(self.R, xk1, jac=self.J, options=self.options)
            xk1 = sol.x
            assert sol.success, f"internal newton method is not converged"

            # Scale ds such that iter goal is satisfied. Disable scaling if we
            # have halved the ds parameter before or after the first iteration
            # which requires lots of iterations see Crisfield1991, section 9.5
            # (9.40) or (9.41) for the square root scaling.
            if self.scale_exponent is not None and sol.nit > 0:
                fac = (self.iter_goal / sol.nit) ** self.scale_exponent
                self.ds *= max(self.MIN_FACTOR, min(fac, self.MAX_FACTOR))

            # store last converged newton step
            self.x0 = self.xk.copy()

            # store new converged newton step
            self.xk = xk1.copy()

            # append solutions to lists
            q_, la_c_, la_g_, la_N_, la_arc_ = np.array_split(xk1, self.split_unknowns)
            q.append(q_)
            la_c.append(la_c_)
            la_g.append(la_g_)
            la_N.append(la_N_)
            la_arc.append(la_arc_[0])

            # update progress bar
            i1 = int(
                100
                * (la_arc_[0] - self.la_arc_span[0])
                / (self.la_arc_span[1] - self.la_arc_span[0])
            )
            pbar.update(i1 - i0)
            pbar.set_description(
                f"la_arc: {self.la_arc_span[0]:0.2e} <= {la_arc_[0]:0.2e} <= {self.la_arc_span[1]:0.2e}; error: {sol.error:0.2e}; iter: {sol.nit}"
            )
            i0 = i1

        # return solution object
        return Solution(
            system=self.system,
            t=np.asarray(la_arc),
            q=np.asarray(q),
            la_c=np.asarray(la_c),
            la_g=np.asarray(la_g),
            la_N=np.asarray(la_N),
        )


class Eigenmodes:
    def __init__(self, system, sol):
        self.system = system
        self.sol = sol

        self.la_sqared_tol = 1e-5

        self.u = np.zeros(system.nu, dtype=float)

        # TODO: it might be benefitial to implement the inverse of c_la_c directly in the contributions
        C = system.c_la_c("csc")
        if system.nla_c > 1:
            self.C_inv = sparse_inv(C)
        else:
            self.C_inv = CooMatrix((system.nla_c, system.nla_c))
            if system.nla_c == 1:
                self.C_inv[0, 0] = 1 / C[0, 0]
            self.C_inv = self.C_inv.asformat("csr")

    def solve(self, index=-1):
        # TODO: check for static equilibrium

        # extract values
        t = self.sol.t[index]
        q = self.sol.q[index]
        la_c = self.sol.la_c[index] if self.sol.la_c is not None else None
        la_g = self.sol.la_g[index] if self.sol.la_g is not None else None
        la_N = self.sol.la_N[index] if self.sol.la_N is not None else None

        ##################
        # stiffness matrix
        ##################
        # Using h, c, g, N contributions for stiffness
        K_h = self.system.KN_h(t, q, self.u)[0]
        K_c = self.system.KN_c(t, q, la_c)[0]
        K_g = self.system.KN_g(t, q, la_g)[0]
        K_N = self.system.KN_N(t, q, la_N)[0]

        # solve compliance equation
        W_c = self.system.W_c(t, q, format="csc")
        K0 = K_h + K_c + K_g + K_N + W_c @ self.C_inv @ W_c.T

        #############
        # mass matrix
        #############
        M0 = self.system.M(t, q)

        #######################
        # bilateral constraints
        #######################
        # TODO: split up in internal and non-internal contributions

        # internal_contr, non_internal_contr = [], []
        # for contr in self.__g_contr:
        #     # TODO: maybe just check if there is a "T" attribute
        #     if hasattr(contr, "nq") and hasattr(contr, "nu"):
        #         internal_contr.append(contr)
        #     else:
        #         non_internal_contr.append(contr)

        # ########################################
        # # A: constraints inside a contribution #
        # ########################################
        # nla_g_intern = int(np.sum([c.nla_g for c in internal_contr]))
        # T_int, col = CooMatrix((self.nu, self.nu - nla_g_intern)), 0
        # removed_laDOFs, changing_uDOFs = [], []
        # for contr in internal_contr:
        #     if hasattr(contr, "T"):
        #         # if there is an implementation
        #         T = contr.T(t, q, format="csc")
        #     else:
        #         # project numerically using W_g
        #         W_g = contr.W_g(t, q[contr.qDOF])
        #         if not isinstance(W_g, np.ndarray):
        #             W_g = W_g.toarray()
        #         T = scipy.sparse.csc_array(scipy.linalg.null_space(W_g.T))

        #     ni_contr = contr.nu - contr.nla_g
        #     T_int[contr.uDOF, col : col + ni_contr] = T
        #     col += ni_contr

        #     removed_laDOFs.extend(contr.la_gDOF)
        #     changing_uDOFs.extend(contr.uDOF)

        # # double check if no wrong DOF was touched
        # removed_laDOFs = np.array(removed_laDOFs)
        # changing_uDOFs = np.array(changing_uDOFs)
        # assert len(removed_laDOFs) == len(
        #     np.unique(removed_laDOFs)
        # ), "Some contributions were working on the same laDOF."
        # assert len(changing_uDOFs) == len(
        #     np.unique(changing_uDOFs)
        # ), "Some contributions were working on the same uDOF."

        # # these are uDOFs by rigid bodies, rods without constraints, ...
        # unchanging_uDOFs = np.setdiff1d(np.arange(self.nu), changing_uDOFs)
        # T_int[unchanging_uDOFs, unchanging_uDOFs] = scipy.sparse.eye_array(
        #     len(unchanging_uDOFs), dtype=float
        # )
        # T_int = T_int.asformat("csc")

        # ######################################
        # # B: remaining bilateral constraints #
        # ######################################
        # # try straight forward Nullspace matrix on W_g.T
        # non_internal_laDOFs = np.setdiff1d(np.arange(self.nla_g), removed_laDOFs)
        # W_g_non_internalT = Wg0[:, non_internal_laDOFs].T.toarray()
        # T_bil = scipy.sparse.csc_array(
        #     scipy.linalg.null_space(W_g_non_internalT @ T_int)
        # )

        W_g = self.system.W_g(t, q, format="csr")
        T = csc_array(null_space(W_g.T.toarray()))

        # T = T_int @ T_bil
        B = self.system.q_dot_u(t, q, format="csc")
        K = T.T @ K0 @ T
        M = T.T @ M0 @ T

        ####################
        # compute eigenmodes
        ####################
        # squared eigenvalues
        res = list(eigh(-K.toarray(), M.toarray()))

        # make everything real
        for i, v in enumerate(res):
            imag_norm = np.linalg.norm(np.imag(v))
            total_norm = np.linalg.norm(v)
            if total_norm > 0.0:
                ratio = imag_norm / total_norm
                if ratio >= 1e-2:
                    print(
                        f"arg(a+bi) = {ratio:.2e}. This imaginary part will be discarded!"
                    )
            res[i] = np.real(v)

        las_ud_squared, Vs_ud = res

        # sort eigenvalues such that rigid body modes are first
        sort_idx = np.argsort(-las_ud_squared)
        las_ud_squared = las_ud_squared[sort_idx]
        Vs_ud = Vs_ud[:, sort_idx]

        # compute omegas
        omegas = np.zeros([len(las_ud_squared)])
        valids = np.ones_like(omegas, dtype=bool)
        modes_dq = B @ T @ Vs_ud
        for i, lai in enumerate(las_ud_squared):
            if np.abs(lai) <= self.la_sqared_tol:
                omegas[i] = 0.0
            elif lai > 0:
                msg = f"Warning: An eigenvalue is larger than 0: lambda = {lai:.3e} --> omega = {np.sqrt(lai):.3e}. This should not happen."
                warnings.warn(msg)
                valids[i] = False
                omegas[i] = np.sqrt(lai)
            else:
                omegas[i] = np.sqrt(-lai)

        # compose solution object with omegas and modes
        sol = Solution(
            self.system,
            np.array([t]),
            np.array([q]),
            omegas=np.array([omegas]),
            modes_dq=np.array([modes_dq]),
            valids=np.array([valids]),
        )

        return omegas, modes_dq, sol
