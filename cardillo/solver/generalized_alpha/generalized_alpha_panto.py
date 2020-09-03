import numpy as np
from scipy.sparse.linalg import spsolve 
from scipy.sparse import csc_matrix, bmat
from tqdm import tqdm

from cardillo.math import Numerical_derivative
from cardillo.solver import Solution

class Generalized_alpha_index3_panto():
    """Generalized alpha solver. 
    Constraints on position level and constraints on velocity level can be solved;
    no derivatives of constraint functions are computed!
    """
    def __init__(self, model, t1, dt, \
                 rho_inf=1, beta=None, gamma=None, alpha_m=None, alpha_f=None,\
                 newton_tol=1e-8, newton_max_iter=40, newton_error_function=lambda x: np.max(np.abs(x))):
        
        self.model = model

        # integration time
        self.t0 = t0 = model.t0
        self.t1 = t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        self.dt = dt

        # parameter
        self.rho_inf = rho_inf
        if None in [beta, gamma, alpha_m, alpha_f]:
            self.alpha_m = (2 * rho_inf - 1) / (1 + rho_inf)
            self.alpha_f = rho_inf / (1 + rho_inf)
            self.gamma = 0.5 + self.alpha_f - self.alpha_m
            self.beta = 0.25 * ((self.gamma + 0.5) ** 2)
        else:
            self.gamma = gamma
            self.beta = beta
            self.alpha_m = alpha_m
            self.alpha_f = alpha_f
        self.alpha_ratio = (1 - self.alpha_f) / (1 - self.alpha_m)

        # newton settings
        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter
        self.newton_error_function = newton_error_function

        # dimensions
        self.nq = model.nq
        self.nu = model.nu
        self.nla_g = model.nla_g

        # equation sof motion, constraints on position level and constraints on velocitiy level
        self.nR = self.nu + self.nla_g

        self.tk = model.t0
        self.qk = model.q0
        self.uk = model.u0
        self.la_gk = model.la_g0

        M0 = model.M(t0, model.q0).tocsr()
        rhs0 = self.model.h(t0, model.q0, model.u0) + self.model.W_g(t0, model.q0) @ model.la_g0
        self.ak = spsolve(M0, rhs0)
        self.a_bark = self.ak.copy()
        
        self.__R_gen = self.__R_gen_analytic

    def update(self, ak1, store=False):
        """update dependent variables modifed version of Capobianco2019 (17):
        - q_dot(uk) instead of uk
        - q_ddot(a_beta) instead of a_beta (weighted a_beta is used inside q_ddot instead of evaluating it twice with both parts)
        - B @ Qk1 instead of Qk1
        """
        dt = self.dt
        dt2 = dt * dt
        a_bark1 = (self.alpha_f * self.ak + (1 - self.alpha_f) * ak1 - self.alpha_m * self.a_bark) / (1 - self.alpha_m)
        uk1 = self.uk + dt * ((1 - self.gamma) * self.a_bark + self.gamma * a_bark1)
        a_beta = (0.5 - self.beta) * self.a_bark + self.beta * a_bark1
        if store:
            self.a_bark = a_bark1
        qk1 = self.qk + dt * self.model.q_dot(self.tk, self.qk, self.uk) + dt2 * self.model.q_ddot(self.tk, self.qk, self.uk, a_beta)
        return qk1, uk1

    def pack(self, a, la_g):
        nu = self.nu
        nla_g = self.nla_g

        x = np.zeros(self.nR)

        x[:nu] = a
        x[nu:nu+nla_g] = la_g

        return x

    def unpack(self, x):
        nu = self.nu
        nla_g = self.nla_g

        # acceleration
        a = x[:nu]

        # constraints on position level
        la_g = x[nu:nu+nla_g]

        return a, la_g
        
    def __R_gen_analytic(self, tk1, xk1):
        nu = self.nu
        nla_g = self.nla_g

        # unpack x and update kinematic variables update dependent variables
        ak1, la_gk1 = self.unpack(xk1)
        qk1, uk1 = self.update(ak1)

        # evaluate mass matrix and constraint force directions and rhs
        Mk1 = self.model.M(tk1, qk1)
        W_gk1 = self.model.W_g(tk1, qk1)
        
        ###################
        # evaluate residual
        ###################
        R = np.zeros(self.nR)

        # equations of motion
        R[:nu] = Mk1 @ ak1 -( self.model.h(tk1, qk1, uk1) + W_gk1 @ la_gk1)

        # constraints on position level
        R[nu:nu+nla_g] = self.model.g(tk1, qk1)
       
        yield R

        ###############################################################################################
        # R[:nu] = Mk1 @ ak1 -( self.model.h(tk1, qk1, uk1) + W_gk1 @ la_gk1 + W_gammak1 @ la_gammak1 )
        ###############################################################################################
        Wla_g_q = self.model.Wla_g_q(tk1, qk1, la_gk1, scipy_matrix=csc_matrix)
        rhs_q = - ( self.model.h_q(tk1, qk1, uk1) + Wla_g_q )
        rhs_u = -self.model.h_u(tk1, qk1, uk1)
        rhs_a = rhs_q @ self.q_a + rhs_u * self.u_a
        Ma_a = self.model.Mu_q(tk1, qk1, ak1, scipy_matrix=csc_matrix) @ self.q_a + rhs_a

        Ra_a = Mk1 + Ma_a
        Ra_la_g = -W_gk1

        #########################################
        # R[nu:nu+nla_g] = self.model.g(tk1, qk1)
        #########################################
        Rla_g_a = self.model.g_q(tk1, qk1) @ self.q_a
        Rla_g_la_g = None
        
        # sparse assemble global tangent matrix
        R_x =  bmat([ [Ra_a,               Ra_la_g],
                      [Rla_g_a,         Rla_g_la_g],
                    ], format='csc')

        yield R_x

        # R_x_num = self.__R_x_num(tk1, xk1)
        # diff = R_x.toarray() - R_x_num

        # # diff_error = diff[:nu] #~1.0e-5

        # # diff_error = diff[nu+nla_g:nu+nla_g+nla_gamma]

        # diff_error = diff #~1.0e-5
        
        # error = np.max(np.abs(diff_error))
        # print(f'absolute error R_x = {error}')

        # # error = np.max(np.abs(diff_error)) / np.max(np.abs(R_x_num))
        # # print(f'relative error R_x = {error}')

        # yield R_x_num
           
    def step(self, tk1, xk1):
        # initial residual and error
        R_gen = self.__R_gen(tk1, xk1)
        R = next(R_gen)
        error = self.newton_error_function(R)
        converged = error < self.newton_tol
        j = 0
        if not converged:
            while j < self.newton_max_iter:
                # jacobian
                R_x = next(R_gen)
                
                # Newton update
                j += 1
                dx = spsolve(R_x, R)
                xk1 -= dx
                R_gen = self.__R_gen(tk1, xk1)
                R = next(R_gen)
                
                error = self.newton_error_function(R)
                converged = error < self.newton_tol
                if converged:
                    break

        return converged, j, error, xk1

    def solve(self): 
        dt = self.dt
        dt2 = self.dt**2

        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        a = [self.ak]
        la_g = [self.la_gk]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            # evaluate quantities at previous time step
            self.q_a = dt2 * self.beta * self.alpha_ratio * self.model.B(self.tk, self.qk, scipy_matrix=csc_matrix)
            self.u_a = dt * self.gamma * self.alpha_ratio

            # initial guess for Newton-Raphson solver and time step
            tk1 = self.tk + self.dt
            xk1 = self.pack(self.ak, self.la_gk)
            converged, n_iter, error, xk1 = self.step(tk1, xk1)
            ak1, la_gk1 = self.unpack(xk1)

            # update progress bar and check convergence
            pbar.set_description(f't: {tk1:0.2e}s < {self.t1:0.2e}s; Newton: {n_iter}/{self.newton_max_iter} iterations; error: {error:0.2e}')
            if not converged:
                raise RuntimeError(f'internal Newton-Raphson method not converged after {n_iter} steps with error: {error:.5e}')

            # update dependent variables
            qk1, uk1 = self.update(ak1, store=True)

            # modify converged quantities
            qk1, uk1 = self.model.solver_step_callback(tk1, qk1, uk1)

            # store soltuion fields
            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            a.append(ak1)

            # update local variables for accepted time step
            self.tk = tk1
            self.qk = qk1
            self.uk = uk1
            self.ak = ak1
            self.la_gk = la_gk1

        # write solution
        return Solution(t=np.array(t), q=np.array(q), u=np.array(u), a=np.array(a), la_g=np.array(la_g))
