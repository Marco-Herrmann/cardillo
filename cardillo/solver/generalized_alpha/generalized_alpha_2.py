import numpy as np
from scipy.sparse.linalg import spsolve 
from scipy.sparse import coo_matrix, csc_matrix, identity, bmat
from tqdm import tqdm

from cardillo.math.prox import prox_Rn0, prox_circle
from cardillo.math import Numerical_derivative
from cardillo.solver import Solution

class Generalized_alpha_2():
    def __init__(self, model, t1, dt, \
                       rho_inf=1, beta=None, gamma=None, alpha_m=None, alpha_f=None,\
                       newton_tol=1e-6, newton_max_iter=100, newton_error_function=lambda x: np.max(np.abs(x)),\
                       numerical_jacobian=False, debug=False):
        
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

        self.beta = 0.25
        self.gamma = 0.5

        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter
        self.newton_error_function = newton_error_function

        self.nq = model.nq
        self.nu = model.nu
        self.nla_g = model.nla_g
        self.nla_gamma = model.nla_gamma
        self.nla_N = model.nla_N
        self.nR = 2 * self.nu + self.nla_g + self.nla_gamma + self.nla_N

        # self.Mk1 = model.M(t0, model.q0)
        # self.W_gk1 = self.model.W_g(t0, model.q0)
        # self.W_gammak1 = self.model.W_gamma(t0, model.q0)
        # self.W_Nk1 = self.model.W_N(t0, model.q0, scipy_matrix=csc_matrix)

        self.tk = model.t0
        self.qk = model.q0 
        self.uk = model.u0 
        self.la_gk = model.la_g0
        self.la_gammak = model.la_gamma0
        self.la_Nk = model.la_N0
        #TODO:
        self.ak = spsolve(model.M(t0, model.q0).tocsr(), self.model.h(t0, model.q0, model.u0) )#+ self.W_gk1 @ model.la_g0 + self.W_gammak1 @ model.la_gamma0 + self.W_Nk1 @ model.la_N0)
        self.Qk = np.zeros(self.nu)

        self.numerical_jacobian = numerical_jacobian
        self.__R_x = self.__R_x_num
        # if numerical_jacobian:
        #     self.__R_x = self.__R_x_num
        # else:
        #     self.__R_x = self.__R_x_analytic
        # self.debug = debug
        # if debug:
        #     self.__R_x = self.__R_x_debug

    def __R(self, tk1, xk1):
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        dt = self.dt
        dt2 = self.dt**2
        ak1 = xk1[:nu]
        Qk1 = xk1[nu:2*nu]
        la_gk1 = xk1[2*nu:2*nu+nla_g]
        la_gammak1 = xk1[2*nu+nla_g:2*nu+nla_g+nla_gamma]
        la_Nk1 = xk1[2*nu+nla_g+nla_gamma:]

        # update dependent variables
        uk1 = self.uk + dt * ((1-self.gamma) * self.ak + self.gamma * ak1) 
        a_beta = (0.5 - self.beta) * self.ak + self.beta * ak1 
        qk1 = self.qk + dt * self.model.q_dot(self.tk, self.qk, self.uk) + dt2 * self.model.q_ddot(self.tk, self.qk, self.uk, a_beta) + self.model.B(self.tk, self.qk) @ Qk1 

        Mk1 = self.model.M(tk1, qk1)
        W_gk1 = self.model.W_g(tk1, qk1)
        W_gammak1 = self.model.W_gamma(tk1, qk1)
        W_Nk1 = self.model.W_N(tk1, qk1, scipy_matrix=csc_matrix)

        # g_N = self.model.g_N(tk1, self.qk)
        # self.I_N = I_N = (g_N <= 0)
        # nI_N = np.count_nonzero(I_N)
        # xi_N = self.model.xi_N(tk1, self.qk1, self.uk, self.uk1)
        # evaluate residual R(ak1, la_gk1, la_gammak1)
        R = np.zeros(self.nR)
        R[:nu] = Mk1 @ ak1 - ( self.model.h(tk1, qk1, uk1) + W_gk1 @ la_gk1 + W_gammak1 @ la_gammak1)# + W_Nk1 @ la_Nk1)
        R[nu:2*nu] = Mk1 @ Qk1 - W_Nk1 @ la_Nk1
        R[2*nu:2*nu+nla_g] = self.model.g(tk1, qk1)
        R[2*nu+nla_g:2*nu+nla_g+nla_gamma] = self.model.gamma(tk1, qk1, uk1)
        R[2*nu+nla_g+nla_gamma:] = la_Nk1 - prox_Rn0(la_Nk1 - self.model.prox_r_N *  self.model.g_N(tk1, qk1))
        return R


    def __R_x_num(self, tk1, xk1):

        R_x_num = Numerical_derivative(self.__R, order=2)._x(tk1, xk1)

        return csc_matrix( R_x_num )
    
    def step(self):
        nu = self.nu
        nla_g = self.nla_g
        nla_gamma = self.nla_gamma
        nla_N = self.nla_N
        dt = self.dt
        tk1 = self.tk + dt

        # initial guess for Newton-Raphson solver
        xk1 = np.zeros(self.nR)
        xk1[:nu] = self.ak
        xk1[nu:2*nu] = self.Qk
        xk1[2*nu:2*nu+nla_g] = self.la_gk
        xk1[2*nu+nla_g:2*nu+nla_g+nla_gamma] = self.la_gammak
        xk1[2*nu+nla_g+nla_gamma:] = self.la_Nk

        # initial residual and error
        R = self.__R(tk1, xk1)
        error = self.newton_error_function(R)
        converged = error < self.newton_tol
        j = 0
        if not converged:
            while j < self.newton_max_iter:
                # jacobian
                R_x = self.__R_x(tk1, xk1)

                # Newton update
                j += 1
                # dx = spsolve(R_x, R)
                try:
                    dx = spsolve(R_x, R)
                except:
                    print('Fehler!!!!')
                xk1 -= dx
                R = self.__R(tk1, xk1)
                error = self.newton_error_function(R)
                converged = error < self.newton_tol
                if converged:
                    break
        ak1 = xk1[:nu]
        Qk1 = xk1[nu:2*nu]
        la_gk1 = xk1[2*nu:2*nu+nla_g]
        la_gammak1 = xk1[2*nu+nla_g:2*nu+nla_g+nla_gamma]
        # la_Nk1 = np.zeros(nla_N)
        # nI_N = np.count_nonzero(self.I_N)
        la_Nk1 = xk1[2*nu+nla_g+nla_gamma:2*nu+nla_g+nla_gamma+nla_N]
        # la_Nk1[~self.I_N] = xk1[nu+nla_g+nla_gamma+nI_N:nu+nla_g+nla_gamma+nla_N] 
            
        return (converged, j, error), tk1, ak1, Qk1, la_gk1, la_gammak1, la_Nk1

    def solve(self): 
        # lists storing output variables
        t = [self.tk]
        q = [self.qk]
        u = [self.uk]
        la_g = [self.la_gk]
        la_gamma = [self.la_gammak]
        la_N = [self.la_Nk]

        pbar = tqdm(np.arange(self.t0, self.t1, self.dt))
        for _ in pbar:
            (converged, n_iter, error), tk1, ak1, Qk1, la_gk1, la_gammak1, la_Nk1 = self.step()
            pbar.set_description(f't: {tk1:0.2e}s < {self.t1:0.2e}s; Newton: {n_iter}/{self.newton_max_iter} iterations; error: {error:0.2e}')
            
            if not converged:
                raise RuntimeError(f'internal Newton-Raphson method not converged after {n_iter} stepts with error: {error:.5e}')
            dt = self.dt
            dt2 = dt * dt
            uk1 = self.uk + dt * ((1-self.gamma) * self.ak + self.gamma * ak1)
            a_beta = (0.5 - self.beta) * self.ak + self.beta * ak1 
            qk1 = self.qk + dt * self.model.q_dot(self.tk, self.qk, self.uk) + dt2 * self.model.q_ddot(self.tk, self.qk, self.uk, a_beta) + self.model.B(self.tk, self.qk) @ Qk1
            qk1, uk1 = self.model.solver_step_callback(tk1, qk1, uk1)
            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            la_g.append(la_gk1)
            la_gamma.append(la_gammak1)
            la_N.append(la_Nk1)

            # update local variables for accepted time step
            self.tk = tk1
            self.qk = qk1
            self.uk = uk1
            self.ak = ak1
            self.Qk = Qk1
            self.la_gk = la_gk1
            self.la_gammak = la_gammak1
            self.la_Nk = la_Nk1

        # write solution
        return Solution(t=np.array(t), q=np.array(q), u=np.array(u), la_g=np.array(la_g), la_gamma=np.array(la_gamma), la_N=np.array(la_N))