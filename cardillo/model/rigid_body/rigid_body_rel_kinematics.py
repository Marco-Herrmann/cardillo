import numpy as np
from cardillo.math.algebra import cross3, ax2skew, A_IK_basic_x, A_IK_basic_y, A_IK_basic_z, dA_IK_basic_x, dA_IK_basic_y, dA_IK_basic_z, inverse3D

from cardillo.math.numerical_derivative import Numerical_derivative

class Rigid_body_rel_kinematics():
    def __init__(self, m, K_theta_S, joint, predecessor, frame_IDp=np.zeros(3), r_OS0=np.zeros(3), A_IK0=np.eye(3)):
        self.m = m
        self.theta = K_theta_S
        self.r_OS0 = r_OS0
        self.A_IK0 = A_IK0

        self.predecessor = predecessor
        self.frame_IDp = frame_IDp
        

        self.joint = joint
        self.nq = joint.get_nq()
        self.nu = joint.get_nu()        
        self.q0 = joint.q0
        self.u0 = joint.u0

        self.is_assembled = False

    def assembler_callback(self):
        if not self.predecessor.is_assembled:
            raise RuntimeError('Predecessor is not assembled.')

        qDOFp = self.predecessor.qDOF_P(self.frame_IDp)
        self.qDOF = np.concatenate([qDOFp, self.qDOF])
        self.nqp = nqp = len(qDOFp)
        self.q0 = np.concatenate([self.predecessor.q0, self.q0])
        self.__nq = nqp + self.nq

        uDOFp = self.predecessor.uDOF_P(self.frame_IDp)
        self.uDOF = np.concatenate([uDOFp, self.uDOF])
        self.nup = nup = len(uDOFp)
        self.u0 = np.concatenate([self.predecessor.u0, self.u0])
        self.__nu = nup + self.nu

        A_IKp = self.predecessor.A_IK(self.predecessor.t0, self.predecessor.q0[qDOFp], self.frame_IDp)
        A_KpB1 = A_IKp.T @ self.joint.A_IB1
        A_KB2 = self.A_IK0.T @ self.joint.A_IB1 @ self.joint.A_B1B2(self.t0, self.q0[nqp:])
        self.A_B2K = A_KB2.T
        r_OSp = self.predecessor.r_OP(self.predecessor.t0, self.predecessor.q0[qDOFp], self.frame_IDp) 
        K_r_SpB1 = A_IKp.T @ (self.joint.r_OB1 - r_OSp)
        self.K_r_SB2 = self.A_IK0.T @ (self.joint.r_OB1 - self.r_OS0)

        self.r_OB1 = lambda t, q: self.predecessor.r_OP(t, q[:nqp], self.frame_IDp, K_r_SpB1)
        self.r_OB1_q1 = lambda t, q: self.predecessor.r_OP_q(t, q[:nqp], self.frame_IDp, K_r_SpB1)
        self.v_B1 = lambda t, q, u: self.predecessor.v_P(t, q[:nqp], u[:nup], self.frame_IDp, K_r_SpB1)
        self.B1_Omegap = lambda t, q, u: A_KpB1.T @ self.predecessor.K_Omega(t, q[:nqp], u[:nup], self.frame_IDp)
        self.B1_Psip = lambda t, q, u, u_dot: A_KpB1.T @ self.predecessor.K_Psi(t, q[:nqp], u[:nup], u_dot[:nup], self.frame_IDp)
        self.a_B1 = lambda t, q, u, u_dot: self.predecessor.a_P(t, q[:nqp], u[:nup], u_dot[:nup], self.frame_IDp, K_r_SpB1)
        # self.J_P1 = lambda t, q: self.subsystem1.J_P(t, q[:nq1], self.frame_ID1, K_r_SP1)
        # self.J_P1_q = lambda t, q: self.subsystem1.J_P_q(t, q[:nq1], self.frame_ID1, K_r_SP1)
        self.A_IB1 = lambda t, q: self.predecessor.A_IK(t, q[:nqp], self.frame_IDp) @ A_KpB1
        self.A_IB1_q1 = lambda t, q: np.einsum('ijl,jk->ikl', self.predecessor.A_IK_q(t, q[:nqp], self.frame_IDp), A_KpB1)
        # self.K_J_R1 = lambda t, q: self.subsystem1.K_J_R(t, q[:nq1], self.frame_ID1)
        # self.K_J_R1_q = lambda t, q: self.subsystem1.K_J_R_q(t, q[:nq1], self.frame_ID1)
        # self.J_R1 = lambda t, q: self.subsystem1.A_IK(t, q[:nq1], self.frame_ID1) @ self.subsystem1.K_J_R(t, q[:nq1], self.frame_ID1)
        # self.J_R1_q = lambda t, q: np.einsum('ijk,jl->ilk', self.subsystem1.A_IK_q(t, q[:nq1], self.frame_ID1), self.subsystem1.K_J_R(t, q[:nq1], self.frame_ID1) ) + np.einsum('ij,jkl->ikl', self.subsystem1.A_IK(t, q[:nq1], self.frame_ID1), self.subsystem1.K_J_R_q(t, q[:nq1], self.frame_ID1) )

        self.A_B1B2 = lambda t, q: self.joint.A_B1B2(t, q[nqp:])
        self.A_B1B2_q2 = lambda t, q: self.joint.A_B1B2_q(t, q[nqp:])

        self.B1_r_B1B2 = lambda t, q: self.joint.B1_r_B1B2(t, q[nqp:])
        self.B1_r_B1B2_q2 = lambda t, q: self.joint.B1_r_B1B2_q(t, q[nqp:])
        self.B1_v_B1B2 = lambda t, q, u: self.joint.B1_v_B1B2(t, q[nqp:], u[nup:])
        self.B1_Omega_B1B2 = lambda t, q, u: self.joint.B1_Omega_B1B2(t, q[nqp:], u[nup:])
        self.B1_a_B1B2 = lambda t, q, u, u_dot: self.joint.B1_a_B1B2(t, q[nqp:], u[nup:], u_dot[nup:])
        self.B1_Psi_B1B2 = lambda t, q, u, u_dot: self.joint.B1_Psi_B1B2(t, q[nqp:], u[nup:], u_dot[nup:])

        self.is_assembled = True

    def M(self, t, q, coo):
        J_S = self.J_P(t, q)
        J_R = self.K_J_R(t, q)
        M = self.m * J_S.T @ J_S + J_R.T @ self.theta @ J_R
        coo.extend(M, (self.uDOF, self.uDOF))

    def f_gyr(self, t, q, u):
        J_R = self.K_J_R(t, q)
        Omega = self.K_Omega(t, q, u)
        return self.m * self.J_P(t, q).T @ self.kappa_P(t, q, u) \
            + self.K_J_R(t, q).T @ (self.theta @ self.K_kappa_R(t, q, u) + cross3(Omega, self.theta @ Omega))

    # def f_gyr_u(self, t, q, u, coo):
    #     omega = u[3:]
    #     dense = np.zeros((self.nu, self.nu))
    #     dense[3:, 3:] = ax2skew(omega) @ self.theta - ax2skew(self.theta @ omega)
    #     coo.extend(dense, (self.uDOF, self.uDOF))

    def q_dot(self, t, q, u):
        q_dot = np.zeros(self.__nq)
        q_dot[:self.nqp] = self.predecessor.q_dot(t, q[:self.nqp], u[:self.nup])
        q_dot[self.nqp:] = self.joint.q_dot(t, q[self.nqp:], u[self.nup:])
        return q_dot
    
    # def q_ddot(self, t, q, u, u_dot):
    #     q_ddot = np.zeros(self.nq)
    #     q_ddot[:3] = u_dot[:3]
    #     q_ddot[3:] = self.Q(q) @ u_dot[3:]

    #     q_dot_q = Numerical_derivative(self.q_dot, order=2)._x(t, q, u)
    #     q_ddot += q_dot_q @ self.q_dot(t, q, u)
    #     return q_ddot

    # def q_dot_q(self, t, q, u, coo):
    #     dense = Numerical_derivative(self.q_dot, order=2)._x(t, q, u)
    #     coo.extend(dense, (self.qDOF, self.qDOF))

    def B(self, t, q, coo):
        coo.extend(self.joint.B(t, q[:self.nqp]), (self.qDOF[self.nqp:], self.uDOF[self.nup:]))

    def qDOF_P(self, frame_ID=None):
        return np.arange(self.__nq)

    def uDOF_P(self, frame_ID=None):
        return np.arange(self.__nu)

    def A_IK(self, t, q, frame_ID=None):
        return self.A_IB1(t, q) @ self.A_B1B2(t, q) @ self.A_B2K

    def A_IK_q(self, t, q, frame_ID=None):
        A_IK_q = np.zeros((3, 3, self.__nq))
        A_IK_q[:, :, :self.nqp] = np.einsum('ijk,jl,lm->imk', self.A_IB1_q1(t, q), self.A_B1B2(t, q), self.A_B2K)
        A_IK_q[:, :, self.nqp:] = np.einsum('ij,jkl,km->iml', self.A_IB1(t, q), self.A_B1B2_q2(t, q), self.A_B2K)
        return A_IK_q

    def r_OP(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.r_OB1(t, q) + self.A_IB1(t, q) @ self.B1_r_B1B2(t, q) + self.A_IK(t, q) @ (K_r_SP - self.K_r_SB2)

    def r_OP_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        r_OP_q = np.einsum('ijk,j->ik', self.A_IK_q(t, q), (K_r_SP - self.K_r_SB2))
        r_OP_q[:, :self.nqp] += self.r_OB1_q1(t, q) + np.einsum('ijk,j->ik', self.A_IB1_q1(t, q), self.B1_r_B1B2(t, q))
        r_OP_q[:, self.nqp:] += self.A_IB1(t, q) @ self.B1_r_B1B2_q2(t, q)
        return r_OP_q

    def v_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        # v_B1 + A_IB1 B1_v_B1B2 + A_IK ( K_Omega x (K_r_SP - self.K_r_SB2) )
        v_B2 = self.v_B1(t, q, u) + self.A_IB1(t, q) @ self.B1_v_B1B2(t, q, u)
        v_B2P = self.A_IK(t, q) @ cross3( self.K_Omega(t, q, u), K_r_SP - self.K_r_SB2 )
        return v_B2 + v_B2P

    def nu_P(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.v_P(t, q, np.zeros(self.__nu), frame_ID=frame_ID, K_r_SP=K_r_SP)

    def a_P(self, t, q, u, u_dot, frame_ID=None, K_r_SP=np.zeros(3)):
        K_r_B2P = K_r_SP - self.K_r_SB2
        a_B2 = self.a_B1(t, q, u, u_dot) + self.A_IB1(t, q) @ self.B1_a_B1B2(t, q, u, u_dot)
        a_B2P = self.A_IK(t, q) @ (cross3(self.K_Psi(t, q, u, u_dot), K_r_B2P) + cross3( self.K_Omega(t, q, u), cross3( self.K_Omega(t, q, u), K_r_B2P )) )
        return a_B2 + a_B2P

    def kappa_P(self, t, q, u, frame_ID=None, K_r_SP=np.zeros(3)):
        return self.a_P(t, q, u, np.zeros(self.__nu), frame_ID=frame_ID, K_r_SP=K_r_SP)

    def J_P(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        J_P = np.zeros((3, self.__nu))
        nu_P = self.nu_P(t, q, frame_ID=frame_ID, K_r_SP=K_r_SP)
        I = np.eye(self.__nu)
        for i in range(self.__nu):
            J_P[:, i] = self.v_P(t, q, I[i], frame_ID=frame_ID, K_r_SP=K_r_SP) - nu_P
        return J_P

    # def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        # J_P_q = np.zeros((3, self.nu, self.nq))
        # J_P_q[:, 3:, :] = np.einsum('ijk,jl->ilk', self.A_IK_q(t, q), -ax2skew(K_r_SP))
        # return J_P_q

    def K_Omega(self, t, q, u, frame_ID=None):
        return (self.B1_Omegap(t, q, u) + self.B1_Omega_B1B2(t, q, u)) @ self.A_B1B2(t, q) @ self.A_B2K

    def K_nu_R(self, t, q, frame_ID=None):
        return self.K_Omega(t, q, np.zeros(self.__nu), frame_ID=frame_ID)

    def K_Psi(self, t, q, u, u_dot, frame_ID=None):
        return (self.B1_Psip(t, q, u, u_dot) + self.B1_Psi_B1B2(t, q, u, u_dot)) @ self.A_B1B2(t, q) @ self.A_B2K

    def K_kappa_R(self, t, q, u, frame_ID=None):
        return self.K_Psi(t, q, u, np.zeros(self.__nu), frame_ID=frame_ID)

    # def K_Omega_q(self, t, q, u, frame_ID=None):
    #     return np.zeros((3, self.nq))

    def K_J_R(self, t, q, frame_ID=None):
        J_R = np.zeros((3, self.__nu))
        nu_R = self.K_nu_R(t, q, frame_ID=frame_ID)
        I = np.eye(self.__nu)
        for i in range(self.__nu):
            J_R[:, i] = self.K_Omega(t, q, I[i], frame_ID=frame_ID) - nu_R
        return J_R

    # def K_J_R_q(self, t, q, frame_ID=None):
        # return np.zeros((3, self.nu, self.nq))