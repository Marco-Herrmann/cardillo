import numpy as np
import warnings
from scipy.sparse import lil_array, bmat, csc_array, eye_array
from scipy.sparse.linalg import inv as sparse_inv
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, null_space, eig, qr, solve
from tqdm import tqdm

from cardillo.math.fsolve import fsolve
from cardillo.solver._base import compute_I_F
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
        t1=1.0,
        verbose=True,
        updated=False,
        options=SolverOptions(),
    ):
        self.system = system
        self.options = options
        self.verbose = verbose
        self.updated = updated
        self.load_steps = np.linspace(system.t0, t1, n_load_steps + 1)
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

        # allocate for coo
        self.jac_coo = CooMatrix((nx, nx))
        self.K_coo = CooMatrix((self.nu, self.nq))
        # fmt: off
        self.W_g_coo = system.W_g(self.load_steps[0], system.q0, format="Coo")
        self.W_c_coo = system.W_c(self.load_steps[0], system.q0, format="Coo")
        self.W_N_coo = system.W_N(self.load_steps[0], system.q0, format="Coo")
        self.h_q_coo = system.h_q(self.load_steps[0], system.q0, self.u0, format="Coo")
        self.Wla_g_q_coo = system.Wla_g_q(self.load_steps[0], system.q0, system.la_g0, format="Coo")
        self.Wla_c_q_coo = system.Wla_c_q(self.load_steps[0], system.q0, system.la_c0, format="Coo")
        self.Wla_N_q_coo = system.Wla_N_q(self.load_steps[0], system.q0, system.la_N0, format="Coo")
        self.g_q_coo = system.g_q(self.load_steps[0], system.q0, format="Coo")
        self.g_S_q_coo = system.g_S_q(self.load_steps[0], system.q0, format="Coo")
        self.c_q_coo = system.c_q(self.load_steps[0], system.q0, self.u0, system.la_c0, format="Coo")
        self.g_N_q_coo = system.g_N_q(self.load_steps[0], system.q0, format="Coo")
        # fmt: on

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
        self.W_g = self.system.W_g(t, q, format="csr", coo=self.W_g_coo)
        self.W_c = self.system.W_c(t, q, format="csr", coo=self.W_c_coo)
        self.W_N = self.system.W_N(t, q, format="csr", coo=self.W_N_coo)
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
        self.K_coo["h_q", :, :] = self.system.h_q(
            t, q, self.u0, format="Coo", coo=self.h_q_coo
        )
        self.K_coo["Wla_g_q", :, :] = self.system.Wla_g_q(
            t, q, la_g, format="Coo", coo=self.Wla_g_q_coo
        )
        self.K_coo["Wla_c_q", :, :] = self.system.Wla_c_q(
            t, q, la_c, format="Coo", coo=self.Wla_c_q_coo
        )
        self.K_coo["Wla_N_q", :, :] = self.system.Wla_N_q(
            t, q, la_N, format="Coo", coo=self.Wla_N_q_coo
        )
        g_q = self.system.g_q(t, q, coo=self.g_q_coo)
        g_S_q = self.system.g_S_q(t, q, coo=self.g_S_q_coo)
        c_q = self.system.c_q(t, q, self.u0, la_c, coo=self.c_q_coo)
        c_la_c = self.system.c_la_c()

        # note: csr_matrix is best for row slicing, see
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array
        g_N_q = self.system.g_N_q(t, q, format="csr", coo=self.g_N_q_coo)

        Rla_N_q = lil_array((self.nla_N, self.nq), dtype=float)
        Rla_N_la_N = lil_array((self.nla_N, self.nla_N), dtype=float)
        for i in range(self.nla_N):
            # TODO: solve dirty fix with 1e-99
            if la_N[i] < self.g_N[i]:
                Rla_N_la_N[i, i] = 1.0
                Rla_N_q[i] = g_N_q[i] * 1e-99
            else:
                Rla_N_la_N[i, i] = 1e-99
                Rla_N_q[i] = g_N_q[i]

        # Rla_N_q = Rla_N_q.toarray()
        # Rla_N_la_N = Rla_N_la_N.toarray()

        # fmt: off
        sf0, sf1, sf2, sf3 = self.split_f
        sx0, sx1, sx2 = self.split_x
        self.jac_coo["K"         ,    :sf0,    :sx0] = self.K_coo
        self.jac_coo["W_g"       ,    :sf0, sx0:sx1] = self.W_g
        self.jac_coo["W_c"       ,    :sf0, sx1:sx2] = self.W_c
        self.jac_coo["W_N"       ,    :sf0, sx2:   ] = self.W_N
        self.jac_coo["g_q"       , sf0:sf1,    :sx0] = g_q
        self.jac_coo["c_q"       , sf1:sf2,    :sx0] = c_q
        self.jac_coo["c_la_c"    , sf1:sf2, sx1:sx2] = c_la_c
        self.jac_coo["Rla_N_q"   , sf2:sf3,    :sx0] = Rla_N_q
        self.jac_coo["Rla_N_la_N", sf2:sf3, sx2:   ] = Rla_N_la_N
        self.jac_coo["g_S_q"     , sf3:   ,    :sx0] = g_S_q
        # fmt: on
        return self.jac_coo.asformat("csc")

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


def null_space_qr(G, tol=1e-12):
    # TODO: handle m=0 case
    m, n = G.shape

    # QR with column pivoting
    Q, R, piv = qr(G, mode="economic", pivoting=True)

    # get rank
    diag = np.abs(np.diag(R))
    rank = np.sum(diag > tol * diag[0])

    if rank != m:
        # raise ValueError("G does not have full rank, there are redundant constraints!")
        print("G does not have full rank, there are redundant constraints!")
    # assert rank == m, "G does not have full rank, there are redundant constraints!"

    dep = piv[:rank]
    free = piv[rank:]

    # G = [G_dep G_free]
    G_dep = G[:, dep]
    G_free = G[:, free]

    # get dependent variables
    X = -solve(G_dep, G_free)

    # build null space
    Z = np.zeros((n, n - rank))

    Z[dep, :] = X
    Z[free, :] = np.eye(n - rank)

    return Z


class Eigenmodes:
    def __init__(self, system, sol, *, g_N_tol=1e-3):
        self.system = system
        self.sol = sol

        self.la_sqared_tol = 1e-5
        self.g_N_tol = g_N_tol

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

    def solve(self, index=-1, *, n_eig=-1, compute_dense=True, verbose=False):
        # TODO: clean up and different bil.constraint levels and contacts!
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
        # TODO: use coo?
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

        n_constraints = (
            self.system.nla_g
            + self.system.nla_gamma
            + self.system.nla_N
            + self.system.nla_F
        )
        if n_constraints == 0:
            T = eye_array(self.system.nu)
            M = M0
            K = K0

        else:
            W_g = self.system.W_g(t, q, format="csr")
            W_gamma = self.system.W_gamma(t, q, format="csr")
            W_N = self.system.W_N(t, q, format="csr")
            W_F = self.system.W_F(t, q, format="csr")

            g_N = self.system.g_N(t, q)
            I_N = g_N <= self.g_N_tol
            I_F = compute_I_F(np.arange(self.system.nla_N)[I_N], self.system)[0]

            W = bmat(
                [
                    [W_g, W_gamma, W_N[:, I_N], W_F[:, I_F]],
                ],
                format="csr",
            )

            # eliminate constraints
            T_svd = csc_array(null_space(W.T.toarray()))  # uses SVD
            T_qr = csc_array(null_space_qr(W.T.toarray()))  # uses qr decomposition
            # TODO: get sparseqer working, but the package is not working with native windows

            # projection
            M_svd = T_svd.T @ M0 @ T_svd
            K_svd = T_svd.T @ K0 @ T_svd

            M_qr = T_qr.T @ M0 @ T_qr
            K_qr = T_qr.T @ K0 @ T_qr

            T = T_svd
            M = M_svd
            K = K_svd

            T = T_qr
            M = M_qr
            K = K_qr

            # T_m bestimmen
            # TODO: get non-massive DOFs from contributions, and perform steps from here on only when necessary
            massive_idx = np.arange(self.system.nu)[M0.sum(axis=0) != 0]
            if len(massive_idx) != self.system.nu:
                Tm = T[massive_idx, :]

                # Nullspace of T_m: reduced directions w/o kin. energy
                # TODO: can we use null_space_qr and reuse Q later?
                N = null_space(Tm.toarray())

                k = N.shape[1]

                if k != 0:
                    print("Performing static condensation")
                    # qr decomposition
                    # TODO: understand what we actually do here
                    # TODO: "sparsify" everything here
                    Q, _ = qr(N, mode="full")

                    Zd = Q[:, k:]  # dynamic coordinates
                    Zs = Q[:, :k]  # static coordinates to be condensed
                    nd = Zd.shape[1]

                    Z = np.hstack((Zd, Zs))

                    # transform
                    # M_eff = (Z.T @ M @ Z)[:nd, :nd]
                    M_eff = Zd.T @ M @ Zd
                    Khat = Z.T @ K @ Z

                    Kdd = Khat[:nd, :nd]
                    Kds = Khat[:nd, nd:]
                    Ksd = Khat[nd:, :nd]
                    Kss = Khat[nd:, nd:]

                    # reduction of DOFs
                    # q = [qd qs], qs = -Kss^{-1} Ksd qd
                    # TODO: this shouldn't be too many DOFs (nd: << :nd), so maybe inverting and multiplying is faster
                    X = np.linalg.solve(Kss, Ksd)
                    C = np.vstack((np.eye(nd), -X))

                    K_eff = Kdd - Kds @ X
                    T_eff = T @ Z @ C

                    # make sparse
                    M = csc_array(M_eff)
                    K = csc_array(K_eff)
                    T = csc_array(T_eff)

            if verbose:
                print(f"""
                    shapes: M0: {M0.shape}, nmassive: {len(massive_idx)}, M_NS: {M_qr.shape}, M: {M.shape}
                    nnz               W_g: {  W_g.nnz:>5}, K0: {   K0.nnz:>5}, M0: {M0.nnz:>5}
                    nnz Null_space_svd: T: {T_svd.nnz:>5},  K: {K_svd.nnz:>5},  M: {M_svd.nnz:>5}
                    nnz Null_space_qr : T: { T_qr.nnz:>5},  K: { K_qr.nnz:>5},  M: { M_qr.nnz:>5}
                    nnz         Final : T: {    T.nnz:>5},  K: {    K.nnz:>5},  M: {    M.nnz:>5}
                """)

        ####################
        # compute eigenmodes
        ####################
        # squared eigenvalues

        if n_eig == -1:
            n_eig = M.shape[0]

        assert (
            0 < n_eig <= M.shape[0]
        ), "n_eig must be between 1 and the maximum number of degrees of freedom of the constrained system after static condensation."

        if compute_dense or n_eig == M.shape[0]:
            res = list(eigh(K.toarray(), M.toarray()))
        else:
            # we want to comp
            # TODO: check which is faster
            # res = list(eigsh(K, k=n_eig, M=M, which="SA"))
            res = list(eigsh(K, k=n_eig, M=M, which="SM"))
            # res = list(eigsh(K, k=n_eig, M=M))

        # make everything real
        # TODO: remove?
        if np.iscomplexobj(res[0]) or np.iscomplexobj(res[1]):
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

        omegas_squared, Vs_ud = res

        # sort eigenvalues such that rigid body modes are first
        sort_idx = np.argsort(omegas_squared)
        omegas_squared = omegas_squared[sort_idx]
        Vs_ud = Vs_ud[:, sort_idx]

        # compute omegas
        omegas = np.zeros([n_eig])
        valids = np.ones_like(omegas, dtype=bool)
        Delta_z = T @ Vs_ud[:, :n_eig]

        for i in range(n_eig):
            omegai2 = omegas_squared[i]
            if np.abs(omegai2) <= self.la_sqared_tol:
                omegas[i] = 0.0
            elif omegai2 < 0:
                om_neg = -np.sqrt(-omegai2)
                msg = f"Warning: omega^2 was negative:{omegai2:.3e}, and will be returned as -sqrt(-omega^2) = {om_neg:.3e}."
                warnings.warn(msg)
                valids[i] = False
                omegas[i] = om_neg
            else:
                omegas[i] = np.sqrt(omegai2)

        # compose solution object with omegas and modes
        sol = Solution(
            self.system,
            t,
            q,
            omegas=omegas,
            Delta_z=Delta_z,
            valids=valids,
        )

        return sol

    def solve_cheap(self, index=-1):

        warnings.warn(
            "Using `solve_cheap` uses the derivatives of the nonlinear equations w.r.t. q and projects with q_dot_u. Furthermore: constraints are eliminated 'to the left and to the right' with different projection matrices. The resulting matrices are not guaranteed to be symmetric, so problems with purely imaginary eigenvales may occur."
        )
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
        K_h = self.system.h_q(t, q, self.u)
        K_c = self.system.Wla_c_q(t, q, la_c)
        K_g = self.system.Wla_g_q(t, q, la_g)
        K_N = self.system.Wla_N_q(t, q, la_N)

        # solve compliance equation
        W_c = self.system.W_c(t, q, format="csc")
        c_q = self.system.c_q(t, q, self.sol.u[index], la_c)
        B = self.system.q_dot_u(t, q, format="csc")
        K0 = -(K_h + K_c + K_g + K_N - W_c @ self.C_inv @ c_q) @ B

        #############
        # mass matrix
        #############
        M0 = self.system.M(t, q)

        W_g = self.system.W_g(t, q, format="csr")
        g_q = self.system.g_q(t, q, format="csr")
        T_left = csc_array(null_space(W_g.T.toarray()))
        T_right = csc_array(null_space((g_q @ B).toarray()))

        K = T_left.T @ K0 @ T_right
        M = T_left.T @ M0 @ T_right

        ####################
        # compute eigenmodes
        ####################
        # squared eigenvalues
        res = list(eig(-K.toarray(), M.toarray()))

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
        # we get the right eigenvectors
        Delta_z = T_right @ Vs_ud
        modes_dq = B @ Delta_z
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
            t,
            q,
            omegas=omegas,
            Delta_z=Delta_z,
            modes_dq=modes_dq,
            valids=valids,
        )

        return sol


class FrequencyResponseFunction:
    def __init__(self, system, sol, *, g_N_tol=1e-3):
        self.system = system
        self.sol = sol

        self.g_N_tol = g_N_tol

        self.u = np.zeros(system.nu, dtype=float)

        # TODO: it might be benefitial to implement the inverse of c_la_c directly in the contributions
        self.C = system.c_la_c("csc")
        if system.nla_c > 1:
            self.C_inv = sparse_inv(self.C)
        else:
            self.C_inv = CooMatrix((system.nla_c, system.nla_c))
            if system.nla_c == 1:
                self.C_inv[0, 0] = 1 / self.C[0, 0]
            self.C_inv = self.C_inv.asformat("csr")

        # system dimensions
        seps = np.cumsum(
            np.array(
                [
                    0,
                    system.nu,
                    system.nu,
                    system.nla_c,
                    system.nla_g,
                ],
                dtype=int,
            )
        )

        self.slices = [slice(seps[i], seps[i + 1]) for i in range(len(seps) - 1)]
        assert self.system.nla_gamma == 0, "No velocity level constraints allowed!"
        self.nx = seps[-1]
        self.nu = self.system.nu
        self.nin = self.system.nin
        self.nout = self.system.nout

        # create coo matrices of linear system
        self.E_coo = CooMatrix((self.nx, self.nx))
        self.A_coo = CooMatrix((self.nx, self.nx))
        self.B_coo = CooMatrix((2 * self.nu, self.nin))
        self.C_coo = CooMatrix((self.nout, 2 * self.nu))
        self.D_coo = CooMatrix((self.nout, self.nin))

        # kinematic equation
        eye_nu = eye_array(self.system.nu)
        self.E_coo["eye_kin", self.slices[0], self.slices[0]] = eye_nu
        self.A_coo["eye_kin", self.slices[0], self.slices[1]] = eye_nu

        # compliance
        self.A_coo["C", self.slices[2], self.slices[2]] = self.C

        # prepare for coo matrices of nonlinear system
        self.KN_h_coo = None
        self.KN_c_coo = None
        self.KN_g_coo = None
        self.KN_N_coo = None

        self.DG_h_coo = None
        self.DG_c_coo = None

        self.W_c_coo = None
        self.W_g_coo = None
        self.W_N_coo = None
        self.W_F_coo = None

    def solve(self, index=-1, s_val=None):
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
        self.KN_h_coo = self.system.KN_h(t, q, self.u, format="Coo", coo=self.KN_h_coo)
        self.KN_c_coo = self.system.KN_c(t, q, la_c, format="Coo", coo=self.KN_c_coo)
        self.KN_g_coo = self.system.KN_g(t, q, la_g, format="Coo", coo=self.KN_g_coo)
        self.KN_N_coo = self.system.KN_N(t, q, la_N, format="Coo", coo=self.KN_N_coo)

        ################
        # damping matrix
        ################
        # Using h, c contributions for damping
        self.DG_h_coo = self.system.DG_h(t, q, self.u, format="Coo", coo=self.DG_h_coo)
        self.DG_c_coo = self.system.DG_c(t, q, la_c, format="Coo", coo=self.DG_c_coo)

        #############
        # mass matrix
        #############
        # TODO: constant mass matrix: evaluate once in init
        M0 = self.system.M(t, q)

        ##############################
        # generalized force directions
        ##############################
        self.W_g_coo = self.system.W_g(t, q, format="Coo", coo=self.W_g_coo)
        self.W_c_coo = self.system.W_c(t, q, format="Coo", coo=self.W_c_coo)
        self.W_N_coo = self.system.W_N(t, q, format="Coo", coo=self.W_N_coo)
        self.W_F_coo = self.system.W_F(t, q, format="Coo", coo=self.W_F_coo)

        # frictional contact
        g_N = self.system.g_N(t, q)
        I_N = g_N <= self.g_N_tol
        I_F = compute_I_F(np.arange(self.system.nla_N)[I_N], self.system)[0]

        W_NF = bmat(
            [
                [self.W_N_coo.tocsc()[:, I_N], self.W_F_coo.tocsc()[:, I_F]],
            ],
            format="csr",
        )

        #####################
        # assemble matrices #
        #####################
        # force equilibrium
        self.E_coo["M", self.slices[1], self.slices[1]] = M0
        self.A_coo["K_h", self.slices[1], self.slices[0]] = -self.KN_h_coo[0]
        self.A_coo["K_c", self.slices[1], self.slices[0]] = -self.KN_c_coo[0]
        self.A_coo["K_g", self.slices[1], self.slices[0]] = -self.KN_g_coo[0]
        self.A_coo["K_N", self.slices[1], self.slices[0]] = -self.KN_N_coo[0]
        self.A_coo["N_h", self.slices[1], self.slices[0]] = -self.KN_h_coo[1]
        self.A_coo["N_c", self.slices[1], self.slices[0]] = -self.KN_c_coo[1]
        self.A_coo["N_g", self.slices[1], self.slices[0]] = -self.KN_g_coo[1]
        self.A_coo["N_N", self.slices[1], self.slices[0]] = -self.KN_N_coo[1]

        self.A_coo["D_h", self.slices[1], self.slices[1]] = -self.DG_h_coo[0]
        self.A_coo["D_c", self.slices[1], self.slices[1]] = -self.DG_c_coo[0]
        self.A_coo["G_h", self.slices[1], self.slices[1]] = -self.DG_h_coo[1]
        self.A_coo["G_c", self.slices[1], self.slices[1]] = -self.DG_c_coo[1]

        self.A_coo["W_c", self.slices[1], self.slices[2]] = self.W_c_coo
        self.A_coo["W_g", self.slices[1], self.slices[3]] = self.W_g_coo

        # compliance
        self.A_coo["WcT", self.slices[2], self.slices[0]] = self.W_c_coo.T

        # constraint
        self.A_coo["W_gT", self.slices[3], self.slices[0]] = self.W_g_coo.T

        # input matrix
        self.B_coo["W_in", self.slices[1], :] = self.system.W_in(t, q)

        # output matrix
        # TODO: put to init
        slice_out = slice(self.slices[0].start, self.slices[1].stop)
        # C_coo["C_out", slice_out, :] = self.system.C_out(t, q)
        # C_coo["C_out", :, self.slices[0]] = self.system.C_out(t, q)
        # TODO: pos and vel?
        self.C_coo[:, self.slices[0]] = self.system.C_out(t, q)

        # throuput matrix
        # self.D_coo[...] = ...

        #################
        # expand matrices with active frictional contacts
        #################
        n_active = W_NF.shape[1]
        zeros_E = lil_array((n_active, n_active), dtype=float)

        W_NF_extended1 = CooMatrix((self.nx, n_active))
        W_NF_extended1[self.slices[1], :] = W_NF
        W_NF_extended1 = W_NF_extended1.tocsc()

        W_NF_extended2 = CooMatrix((self.nx, n_active))
        W_NF_extended2[self.slices[0], :] = W_NF
        W_NF_extended2 = W_NF_extended2.tocsc()

        E = bmat(
            [
                [self.E_coo.tocsc(), None],
                [None, zeros_E],
            ]
        )
        A = bmat(
            [
                [self.A_coo.tocsc(), W_NF_extended1],
                [W_NF_extended2.T, None],
            ]
        )

        # TODO: use coo?
        # too sparse
        E = E.asformat("csc")
        A = A.asformat("csc")
        B = self.B_coo.asformat("csc")
        C = self.C_coo.asformat("csc")
        D = self.D_coo.asformat("csc")

        H = lambda s: C @ sparse_inv(E * s - A)[: 2 * self.nu, : 2 * self.nu] @ B + D
        if s_val is None:
            return H

        return np.array([H(si).todense() for si in s_val])
