import numpy as np

from cardillo.utility.coo import Coo
from cardillo.discretization import gauss
from cardillo.discretization import uniform_knot_vector, B_spline_basis
from cardillo.discretization import Lagrange_basis
from cardillo.math.algebra import norm2, norm3
from cardillo.math.numerical_derivative import Numerical_derivative

class Rope(object):
    def __init__(self, A_rho0, material_model, polynomial_degree, nEl, nQP, Q=None, q0=None, u0=None, B_splines=True, dim=3):
        self.dim = dim
        if dim == 2:
            self.norm = norm2
        elif dim == 3:
            self.norm = norm3
        else:
            raise ValueError('dim has to be 2 or 3')
        
        # physical parameters
        self.A_rho0 = A_rho0

        # material model
        self.material_model = material_model

        # discretization parameters
        self.B_splines = B_splines
        self.polynomial_degree = polynomial_degree # polynomial degree
        self.nQP = nQP # number of quadrature points
        self.nEl = nEl # number of elements

        if B_splines:
            nn = nEl + polynomial_degree # number of nodes
            self.knot_vector = knot_vector = uniform_knot_vector(polynomial_degree, nEl) # uniform open knot vector
            self.element_span = np.asarray( self.knot_vector[polynomial_degree:-polynomial_degree] )
        else:
            nn = nEl * polynomial_degree + 1 # number of nodes
            self.element_span = np.linspace(0, 1, nEl + 1)

        nn_el = polynomial_degree + 1 # number of nodes per element
        nq_n = dim # number of degrees of freedom per node

        self.nq = nn * nq_n # total number of generalized coordinates
        self.nu = self.nq
        self.nq_el = nn_el * nq_n # total number of generalized coordinates per element

        # compute allocation matrix
        if B_splines:
            self.basis_functions = self.__basis_functions_b_splines
            row_offset = np.arange(nEl)
            elDOF_row = (np.zeros((nq_n * nn_el, nEl), dtype=int) + row_offset).T
            elDOF_tile = np.tile(np.arange(0, nn_el), nq_n)
            elDOF_repeat = np.repeat(np.arange(0, nq_n * nn, step=nn), nn_el)
            self.elDOF = elDOF_row + elDOF_tile + elDOF_repeat
        else:
            self.basis_functions = self.__basis_functions_lagrange
            row_offset = np.arange(0, nn - polynomial_degree, polynomial_degree)
            elDOF_row = (np.zeros((nq_n * nn_el, nEl), dtype=int) + row_offset).T
            elDOF_tile = np.tile(np.arange(0, nn_el), nq_n)
            elDOF_repeat = np.repeat(np.arange(0, nq_n * nn, step=nn), nn_el)
            self.elDOF = elDOF_row + elDOF_tile + elDOF_repeat
            
        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q
        self.q0 = q0
        self.u0 = u0

        # compute shape functions
        derivative_order = 1
        self.N  = np.empty((nEl, nQP, nn_el))
        self.N_xi = np.empty((nEl, nQP, nn_el))
        self.qw = np.zeros((nEl, nQP))
        self.xi = np.zeros((nEl, nQP))
        self.J0 = np.zeros((nEl, nQP))
        for el in range(nEl):
            if B_splines:
                # evaluate Gauss points and weights on [xi^el, xi^{el+1}]
                qp, qw = gauss(nQP, self.element_span[el:el+2])

                # store quadrature points and weights
                self.qw[el] = qw
                self.xi[el] = qp

                # evaluate B-spline shape functions
                N_dN = B_spline_basis(polynomial_degree, derivative_order, knot_vector, qp)
                self.N[el] = N_dN[:, 0]
                self.N_xi[el] = N_dN[:, 1]
            else:
                # evaluate Gauss points and weights on [-1, 1]
                qp, qw = gauss(nQP)

                # store quadrature points and weights
                self.qw[el] = qw
                diff_xi = self.element_span[el + 1] - self.element_span[el]
                sum_xi = self.element_span[el + 1] + self.element_span[el]
                self.xi[el] = diff_xi * qp  / 2 + sum_xi / 2
                
                self.N[el], self.N_xi[el] = Lagrange_basis(polynomial_degree, qp, derivative=True)

            # compute change of integral measures
            Qe = self.Q[self.elDOF[el]]
            for i in range(nQP):
                r0_xi = np.kron(np.eye(self.dim), self.N_xi[el, i]) @ Qe
                self.J0[el, i] = self.norm(r0_xi)

        # shape functions on the boundary
        N_bdry = np.zeros(nn_el)
        N_bdry[0] = 1
        N_bdry_left = np.kron(np.eye(dim), N_bdry)

        N_bdry = np.zeros(nn_el)
        N_bdry[-1] = 1
        N_bdry_right = np.kron(np.eye(dim), N_bdry)

        self.N_bdry = np.array([N_bdry_left, N_bdry_right])

    def __basis_functions_b_splines(self, xi):
        return B_spline_basis(self.polynomial_degree, 0, self.knot_vector, xi)

    def __basis_functions_lagrange(self, xi):
        el = np.where(xi >= self.element_span)[0][-1]
        diff_xi = self.element_span[el + 1] - self.element_span[el]
        sum_xi = self.element_span[el + 1] + self.element_span[el]
        xi_tilde = (2 * xi - sum_xi) / diff_xi
        return Lagrange_basis(self.polynomial_degree, xi_tilde, derivative=False)

    def assembler_callback(self):
        self.__M_coo()

    #########################################
    # equations of motion
    #########################################
    def M_el(self, N, J0, qw):
        Me = np.zeros((self.nq_el, self.nq_el))

        for Ni, J0i, qwi in zip(N, J0, qw):
            # build matrix of shape functions and derivatives
            NNi = np.kron(np.eye(self.dim), Ni)
            
            # integrate elemente mass matrix
            Me += NNi.T @ NNi * self.A_rho0 * J0i * qwi

        return Me
    
    def __M_coo(self):
        self.__M = Coo((self.nu, self.nu))
        for el in range(self.nEl):
            # extract element degrees of freedom
            elDOF = self.elDOF[el, :]

            # compute element mass matrix
            Me = self.M_el(self.N[el], self.J0[el], self.qw[el])
            
            # sparse assemble element mass matrix
            self.__M.extend(Me, (self.uDOF[elDOF], self.uDOF[elDOF]))

    def M(self, t, q, coo):
        coo.extend_sparse(self.__M)

    def E_pot_el(self, qe, N_xi, J0, qw):
        E_pot = 0

        for N_xii, J0i, qwi in zip(N_xi, J0, qw):
            # build matrix of shape function derivatives
            NN_xii = np.kron(np.eye(self.dim), N_xii)

            # tangential vectors
            dr  = NN_xii @ qe 
            g = self.norm(dr)
            
            # Calculate the strain and stress
            lambda_ = g / J0i
            stress = self.material_model.n(lambda_) * dr / g

            # integrate element force vector
            # fe -= (dr / J0i) @ stress * J0i * qwi
            E_pot -= dr @ stress * qwi

        return E_pot
    
    def E_pot(self, t, q):
        E_pot = 0
        for el in range(self.nEl):
            E_pot += self.E_pot_el(q[self.elDOF[el]], self.N_xi[el], self.J0[el], self.qw[el])     
        return E_pot

    def f_pot_el(self, qe, N_xi, J0, qw):
        fe = np.zeros(self.nq_el)

        for N_xii, J0i, qwi in zip(N_xi, J0, qw):
            # build matrix of shape function derivatives
            NN_xii = np.kron(np.eye(self.dim), N_xii)

            # tangential vectors
            dr  = NN_xii @ qe 
            g = self.norm(dr)
            
            # Calculate the strain and stress
            lambda_ = g / J0i
            stress = self.material_model.n(lambda_) * dr / g

            # integrate element force vector
            # fe -= (NN_xii.T / J0i) @ stress * J0i * qwi
            fe -= NN_xii.T @ stress * qwi

        return fe
    
    def f_pot(self, t, q):
        f = np.zeros(self.nu)
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            f[elDOF] += self.f_pot_el(q[elDOF], self.N_xi[el], self.J0[el], self.qw[el])
        return f

    def f_pot_q_el(self, qe, N_xi, J0, qw):        
        fe_q = np.zeros((self.nq_el, self.nq_el))

        for N_xii, J0i, qwi in zip(N_xi, J0, qw):
            # build matrix of shape function derivatives
            NN_xii = np.kron(np.eye(self.dim), N_xii)

            # tangential vectors
            dr  = NN_xii @ qe 
            g = self.norm(dr)
            
            # Calculate the strain and stress
            lambda_ = g / J0i
            n = self.material_model.n(lambda_)
            dn = self.material_model.dn(lambda_)
            dstress = NN_xii / g * n + np.outer(dr, dr) @ NN_xii / g**2 * (dn / J0i - n / g)

            # Calcualte element stiffness matrix
            # fe_q -= (NN_xii.T / J0i) @ dstress * J0i * qwi
            fe_q -= NN_xii.T @ dstress * qwi

        # fe_q_num = Numerical_derivative(lambda t, qe: self.f_pot_el(qe, N_xi, J0, qw), order=2)._x(0, qe, eps=1.0e-6)
        # diff = fe_q_num - fe_q
        # error = np.linalg.norm(diff)
        # print(f'error in stiffness matrix: {error:.4e}')
        # return fe_q_num

        return fe_q

    def f_pot_q(self, t, q, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el, :]
            Ke = self.f_pot_q_el(q[elDOF], self.N_xi[el], self.J0[el], self.qw[el])

            # sparse assemble element internal stiffness matrix
            coo.extend(Ke, (self.uDOF[elDOF], self.qDOF[elDOF]))

    #########################################
    # kinematic equation
    #########################################
    def q_dot(self, t, q, u):
        return u

    def B(self, t, q, coo):
        coo.extend_diag(np.ones(self.nq), (self.qDOF, self.uDOF))

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    ####################################################
    # interactions with other bodies and the environment
    ####################################################
    def elDOF_P(self, frame_ID):
        xi = frame_ID[0]
        if xi == 0:
            return self.elDOF[0]
        elif xi == 1:
            return self.elDOF[-1]
        else:
            el = np.where(xi >= self.element_span)[0][-1]
            return self.elDOF[el]

    def qDOF_P(self, frame_ID):
        return self.elDOF_P(frame_ID)

    def uDOF_P(self, frame_ID):
        return self.elDOF_P(frame_ID)

    def r_OP(self, t, q, frame_ID, K_r_SP=None):
        return self.r_OP_q(t, q, frame_ID) @ q

    def r_OP_q(self, t, q, frame_ID, K_r_SP=None):
        xi = frame_ID[0]
        if xi == 0:
            NN = self.N_bdry[0]
        elif xi == 1:
            NN = self.N_bdry[1]
        else:
            NN = np.kron(np.eye(self.dim), self.basis_functions(xi))

        # interpolate position vector
        r_q = np.zeros((3, self.nq_el))
        r_q[:self.dim] = NN
        return r_q

    def v_P(self, t, q, u, frame_ID, K_r_SP=None):
        return self.r_OP(t, u, frame_ID=frame_ID)

    def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        return self.r_OP(t, u_dot, frame_ID=frame_ID)

    def v_P_q(self, t, q, u, frame_ID, K_r_SP=None):
        return self.r_OP_q(t, None, frame_ID=frame_ID)

    def J_P(self, t, q, frame_ID, K_r_SP=None):
        return self.r_OP_q(t, None, frame_ID=frame_ID)

    def J_P_q(self, t, q, frame_ID, K_r_SP=None):
        return np.zeros((3, self.nq_el, self.nq_el))

    def body_force_pot_el(self, force, t, qe, N, xi, J0, qw):
        E_pot = 0
        for Ni, xii, J0i, qwi in zip(N, xi, J0, qw):
            NNi = np.kron(np.eye(self.dim), Ni)
            E_pot -= (NNi @ qe) @ force(xii, t) * J0i * qwi
        return E_pot

    def body_force_pot(self, t, q, force):
        E_pot = 0
        for el in range(self.nEl):
            E_pot += self.body_force_pot_el(force, t, q[self.elDOF[el]], self.N[el], self.xi[el], self.J0[el], self.qw[el])
        return E_pot

    def body_force_el(self, force, t, N, xi, J0, qw):
        fe = np.zeros(self.nq_el)

        for Ni, xii, J0i, qwi in zip(N, xi, J0, qw):
            NNi = np.kron(np.eye(self.dim), Ni)
            fe += NNi.T @ force(xii, t) * J0i * qwi
        
        return fe

    def body_force(self, t, q, force):
        f = np.zeros(self.nq)

        for el in range(self.nEl):
            f[self.elDOF[el, :]] += self.body_force_el(force, t, self.N[el], self.xi[el], self.J0[el], self.qw[el])
        
        return f

    def body_force_q(self, t, q, coo, force):
        pass