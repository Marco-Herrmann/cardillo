import numpy as np
from cardillo.math import Exp_SO3, Log_SO3, T_SO3, approx_fprime
from cardillo.beams._base import TimoshenkoPetrovGalerkinBase


class Crisfield1999(TimoshenkoPetrovGalerkinBase):
    def __init__(
        self,
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        polynomial_degree_r,
        polynomial_degree_psi,
        nelement,
        Q,
        q0=None,
        u0=None,
        basis_r="Lagrange",
        basis_psi="Lagrange",
    ):
        # reference rotation for relative rotation proposed by Crisfield1999 (5.8)
        nnodes_element_psi = polynomial_degree_psi + 1
        self.node_A = int(0.5 * (nnodes_element_psi + 1)) - 1
        self.node_B = int(0.5 * (nnodes_element_psi + 2)) - 1

        super().__init__(
            cross_section,
            material_model,
            A_rho0,
            K_S_rho0,
            K_I_rho0,
            polynomial_degree_r,
            polynomial_degree_psi,
            nelement,
            Q,
            q0=q0,
            u0=u0,
            basis_r=basis_r,
            basis_psi=basis_psi,
        )

    def _reference_rotation(self, qe: np.ndarray):
        """Reference rotation proposed by Crisfield1999 (5.8)."""
        A_0I = Exp_SO3(qe[self.nodalDOF_element_psi[self.node_A]])
        A_0J = Exp_SO3(qe[self.nodalDOF_element_psi[self.node_B]])
        A_IJ = A_0I.T @ A_0J  # Crisfield1999 (5.8)
        phi_IJ = Log_SO3(A_IJ)
        return A_0I @ Exp_SO3(0.5 * phi_IJ)

    def _relative_interpolation(
        self, A_IR: np.ndarray, qe: np.ndarray, N_psi: np.ndarray, N_psi_xi: np.ndarray
    ):
        """Interpolation function for relative rotation vectors proposed by
        Crisfield1999 (5.7) and (5.8)."""
        # relative interpolation of local rotation vectors
        psi_rel = np.zeros(3, qe.dtype)
        psi_rel_xi = np.zeros(3, qe.dtype)
        for node in range(self.nnodes_element_psi):
            # nodal axis angle vector
            psi_node = qe[self.nodalDOF_element_psi[node]]

            # nodal rotation
            A_IK_node = Exp_SO3(psi_node)

            # relative rotation of each node and corresponding
            # rotation vector
            A_RK_node = A_IR.T @ A_IK_node
            psi_RK_node = Log_SO3(A_RK_node)

            # add wheighted contribution of local rotation
            psi_rel += N_psi[node] * psi_RK_node
            psi_rel_xi += N_psi_xi[node] * psi_RK_node

        return psi_rel, psi_rel_xi

    def _eval(self, qe, xi):
        # evaluate shape functions
        N_r, N_r_xi = self.basis_functions_r(xi)
        N_psi, N_psi_xi = self.basis_functions_psi(xi)

        # interpolate tangent vector
        r_OP = np.zeros(3, dtype=qe.dtype)
        r_OP_xi = np.zeros(3, dtype=qe.dtype)
        for node in range(self.nnodes_element_r):
            r_OP_node = qe[self.nodalDOF_element_r[node]]
            r_OP += N_r[node] * r_OP_node
            r_OP_xi += N_r_xi[node] * r_OP_node

        # reference rotation, see. Crisfield1999 (5.8)
        A_IR = self._reference_rotation(qe)

        # relative interpolation of the rotation vector and it first derivative
        psi_rel, psi_rel_xi = self._relative_interpolation(A_IR, qe, N_psi, N_psi_xi)

        # objective rotation
        A_IK = A_IR @ Exp_SO3(psi_rel)

        # axial and shear strains
        K_Gamma_bar = A_IK.T @ r_OP_xi

        # objective curvature
        T = T_SO3(psi_rel)
        K_Kappa_bar = T @ psi_rel_xi

        return r_OP, A_IK, K_Gamma_bar, K_Kappa_bar

    def A_IK(self, t, q, frame_ID):
        return self._eval(q, frame_ID[0])[1]
        # evaluate shape functions
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # interpolate orientation
        A_IK = np.zeros((3, 3), dtype=q.dtype)
        for node in range(self.nnodes_element_psi):
            A_IK += N_psi[node] * Exp_SO3(q[self.nodalDOF_element_psi[node]])

        return A_IK

    def A_IK_q(self, t, q, frame_ID):
        return approx_fprime(q, lambda q: self.A_IK(t, q, frame_ID))

        # evaluate shape functions
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # interpolate centerline position and orientation
        A_IK_q = np.zeros((3, 3, self.nq_element), dtype=q.dtype)
        for node in range(self.nnodes_element_psi):
            nodalDOF_psi = self.nodalDOF_element_psi[node]
            A_IK_q[:, :, nodalDOF_psi] += N_psi[node] * Exp_SO3_psi(q[nodalDOF_psi])

        return A_IK_q