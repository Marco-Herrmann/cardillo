import numpy as np
from scipy.sparse.linalg import spsolve 
from scipy.sparse import csr_matrix

class Euler_backward():
    r""" Euler backward

    Parameters
    ----------
    model : Model
        Mechanical model
    t_span : list, tuple or numpy.ndarray
        integration domain, t_span[0] and t_span[-1] are the start and end integration time, respetively; if t_span contains only two elements no dense output will be generated, otherwise the given points are used for computing a dense output
    dt : float, optional
        user given time step; if variable_step_size is chosen this is the initial step size; defualt value is None, then we use a conservative initial value as given in :cite:`Hairer1993` p. 169
    atol : float or numpy.ndarray, optional
        Absolute tolerance used for the error estimation. Let $y_{1i}$ and $\\hat{y}_{1i}$ being the $i$-th components of the solution at the next time step, computed by the generalized-$\\alpha$ scheme and the backward Euler method, respectively. The last converged solution of the generalized-$\\alpha$ method is $y_{0i}$. We estimate the error using $e = \\sqrt{\\frac{1}{n} \\sum_{i=1}^n \\left(\\frac{y_{1i} - \\hat{y}_{1i}}{sc_i}\\right)^2}$ with $sc_i = atol_i + rtol_i~\\mathrm{max}(\\lvert y_{0i}\\rvert, \\lvert y_{1i} \\rvert)$. Where $atol_i$ and $rtol_i$ are the desired tolerances prescribed by the user (relative errors are considered for $atol_i = 0$, absolute errors for $rtol_i = 0$; usually both tolerances are different from zero.
    rtol : float or numpy.ndarray, optional
        Relative tolerance used for the error estimation. Let $y_{1i}$ and $\\hat{y}_{1i}$ being the $i$-th components of the solution at the next time step, computed by the generalized-$\\alpha$ scheme and the backward Euler method, respectively. The last converged solution of the generalized-$\\alpha$ method is $y_{0i}$. We estimate the error using $e = \\sqrt{\\frac{1}{n} \\sum_{i=1}^n \\left(\\frac{y_{1i} - \\hat{y}_{1i}}{sc_i}\\right)^2}$ with $sc_i = atol_i + rtol_i~\\mathrm{max}(\\lvert y_{0i}\\rvert, \\lvert y_{1i} \\rvert)$. Where $atol_i$ and $rtol_i$ are the desired tolerances prescribed by the user (relative errors are considered for $atol_i = 0$, absolute errors for $rtol_i = 0$; usually both tolerances are different from zero.
    newton_max_iter : int, optional
        maximum number of iterations for internal Newton--Raphson steps.
    newton_tol : float, optional
        tolerance for internal Newton--Raphson steps.
    newton_error_function : :ref:`lambda<python:lambda>` with numpy.ndarray as argument, optional
        Function which is used for computing the error in the underlying Newton-Raphson method. The maximum absolute value is the default funciton.

    Notes
    -----
    

    Note
    ---- 
    """
    def __init__(self, model, t_span, dt, newton_tol=1e-6, newton_max_iter=10, newton_error_function=lambda x: np.max(np.abs(x))):
        
        self.model = model

        # integration time
        self.t_span = np.asarray(t_span)
        if self.t_span.ndim != 1:
            raise ValueError("`t_span` must be 1-dimensional.")
        d = np.diff(self.t_span)
        if np.any(d <= 0):
            raise ValueError("Values in `t_span` are not an increasing sequence.")
        self.t0, self.t1 = self.t_span[[0, -1]]
        
        # constant time step
        self.dt = dt

        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter
        self.newton_error_function = newton_error_function
        self.linearSolver = spsolve

        self.nq = self.model.nq
        self.nu = self.model.nu
        self.nla_g = self.model.nla_g
        self.n = self.nq + self.nu + self.nla_g

        self.uDOF = np.arange(self.nu)
        self.qDOF = self.nu + np.arange(self.nq)
        self.la_gDOF = self.nu + self.nq + np.arange(self.nla_g)

    def __R(self, qk, uk, tk1, qk1, uk1, la_gk1):
        R = np.zeros(self.n)

        R[self.uDOF] = self.model.M(tk1, qk1, scipy_matrix=csr_matrix) @ (uk1 - uk) - self.dt * (self.model.h_u(tk1, qk1, uk1) + self.model.Wla_g(tk1, qk1, la_gk1))
        R[self.qDOF] = qk1 - qk - self.dt * self.model.q_dot(tk1, qk1, uk1)
        R[self.la_gDOF] = self.model.g(tk1, qk1)

    def __dR(self, qk, uk, tk1, qk1, uk1, la_gk1):
        dR = np.zeros((self.n, self.n))

        dR_uu = self.model.M(tk1, qk1) - self.dt * self.model.h_u(tk1, qk1, uk1)
        dR[self.nu:self.nu+self.nq] = qk1 - qk - self.dt * self.model.q_dot(tk1, qk1, uk1)
        dR[self.nu+self.nq:] = self.model.g(tk1, qk1)

    def step(self, tk, qk, uk):
        # general quantities
        dt = self.dt

        tk1 = tk + dt
        uk1 = uk + dt * self.model.u_dot(tk, qk, uk)
        qk1 = qk + dt * self.model.q_dot(tk, qk, uk)
        
        return tk1, qk1, uk1

    def solve(self): 
        
        # lists storing output variables
        tk = self.t0
        qk = self.model.q0.copy()
        uk = self.model.u0.copy()
        
        t = [tk]
        q = [qk]
        u = [uk]

        while tk <= self.t1:
            tk1, qk1, uk1 = self.step(tk, qk, uk)

            qk1, uk1 = self.model.callback(tk1, qk1, uk1)

            t.append(tk1)
            q.append(qk1)
            u.append(uk1)
            # update local variables for accepted time step
            tk, qk, uk = tk1, qk1, uk1
            
        # write solution
        return np.array(t), np.array(q), np.array(u)
    
    