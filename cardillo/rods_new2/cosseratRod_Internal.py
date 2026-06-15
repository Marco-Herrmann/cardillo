from abc import ABC, abstractmethod
import numpy as np
from cachetools import cachedmethod, LRUCache
from cachetools.keys import hashkey
from scipy.sparse import (
    block_diag,
    bsr_array,
    csr_array,
    eye_array,
)
from scipy.sparse.linalg import spsolve
from warnings import warn

from cardillo.math.algebra import norm, cross3, ax2skew, ax2skew_a
from cardillo.math.approx_fprime import approx_fprime
from cardillo.math.rotations import (
    Log_SO3_quat,
    Exp_SO3_quat,
    Exp_SO3_quat_P,
    T_SO3_quat,
    T_SO3_quat_P,
    T_SO3_inv_quat,
    T_SO3_inv_quat_P,
    Log_SO3_R9,
    Exp_SO3_R9,
    Exp_SO3_R9_R9,
    T_SO3_R9,
    T_SO3_R9_R9,
    T_SO3_inv_R9,
    T_SO3_inv_R9_R9,
)
from cardillo.utility.coo_matrix import CooMatrix
from cardillo.utility.sparse_array_blocks import SparseArrayBlocks

zeros3 = np.zeros(3, dtype=float)
eye3 = np.eye(3, dtype=float)


class CosseratRod_Internal:
    def __init__(self, parent, mesh_kin, mesh_cg, quadrature):
        self.parent = parent
        self.nnodes_sigma = self.parent.nnodes_sigma

        # quadrature
        quadrature_int_kin = mesh_kin.quadrature(*quadrature, 1)
        self.nquadrature_int_total = quadrature_int_kin["nquadrature_total"]
        self.qp_int_vec = quadrature_int_kin["qp"]
        self.qw_int_vec = quadrature_int_kin["qw"]
        self.qels_int_vec = quadrature_int_kin["els"]
        self.N_int, self.N_xi_int = quadrature_int_kin["N"]

        quadrature_int_cg = mesh_cg.quadrature(*quadrature, 0)
        self.Nc_int = quadrature_int_cg["N"][0]

        # handle stuff
        self.idx_c = parent.idx_c
        self.idx_g = parent.idx_g
        self.idx_db = parent.idx_db
        self.nla_sigma = mesh_cg.nnodes * 6
        nla_c = mesh_cg.nnodes * len(self.idx_c)
        nla_g = mesh_cg.nnodes * len(self.idx_g)

        # compliance contributions
        self._nla_c = nla_c
        if nla_c > 0:
            self.parent.nla_c = nla_c

            # c = c_la_c @ la_c - l_c
            # --> la_c = -c_la_c_inv @ c(q, u, 0) = c_la_c_inv @ l_c(q)
            self.parent.la_c = lambda t, q, u: self.c_la_c_inv @ self.l_sigma(q)[0]
            self.parent.c = (
                lambda t, q, u, la_c: self.c_la_c @ la_c - self.l_sigma(q)[0]
            )
            self.parent.c_q = lambda t, q, u, la_c: -self.l_sigma_q(q)[0]
            self.parent.c_la_c = lambda: self.c_la_c
            self.parent.W_c = lambda t, q: self.W_sigma(q)[0]
            self.parent.Wla_c_q = lambda t, q, la_c: self.Wla_sigma_q(q, la_c, None)
            self.parent.KN_c = lambda t, q, la_c: (self.K_sigma(q, la_c, None), None)
            self.parent.E_pot_comp = self.E_pot_comp

        # constraint
        self._nla_g = nla_g
        self.include_g = nla_g > 0

        # displacement-based
        self._nDB = len(self.idx_db)
        self.include_f_pot = self._nDB > 0

    def set_material_model(self, material_model):
        self.material_model = material_model
        self.material_model_qp = self.material_model.prepare(self.qp_int_vec)

    def set_reference_strains(self, Q):
        _, B_gamma0_bar, B_kappa0_bar = self.parent.kinematics._eval_internal_vec(
            self.N_int, self.N_xi_int, Q
        )
        self.J_int_vec = np.linalg.norm(B_gamma0_bar, axis=1)
        self.B_gamma0_bar_int = B_gamma0_bar
        self.B_kappa0_bar_int = B_kappa0_bar
        self.epsilon0_int = (
            np.hstack([B_gamma0_bar, B_kappa0_bar]) / self.J_int_vec[:, None]
        )

        if self._nla_c > 0:
            # c_la_c
            c_la_c_pairs = [
                (self.Nc_int, self.Nc_int, self.qw_int_vec * self.J_int_vec)
            ]
            self.c_la_c_SAB = SparseArrayBlocks(
                (self._nla_c, self._nla_c),
                (len(self.idx_c), len(self.idx_c)),
                c_la_c_pairs,
            )

        # c_sigma
        c_sigma_q_pairs = [
            (self.Nc_int, self.N_int, self.qw_int_vec),
            (self.Nc_int, self.N_xi_int, self.qw_int_vec),
        ]
        self.c_sigma_q_SAB = SparseArrayBlocks(
            (self.nla_sigma, self.parent.nq),
            (6, self.parent.kinematics.nq_node),
            c_sigma_q_pairs,
            [(self.idx_c, ...), (self.idx_g, ...)],
        )

        # W_sigma
        W_sigma_pairs = [
            (self.N_int, self.Nc_int, self.qw_int_vec),
            (self.N_xi_int, self.Nc_int, self.qw_int_vec),
        ]
        self.W_sigma_SAB = SparseArrayBlocks(
            (self.parent.nu, self.nla_sigma),
            (6, 6),
            W_sigma_pairs,
            [(..., self.idx_c), (..., self.idx_g)],
        )

        # Wla_sigma_q
        h_pot_q_pairs = [
            (self.N_int, self.N_int, self.qw_int_vec),
            (self.N_int, self.N_xi_int, self.qw_int_vec),
            (self.N_xi_int, self.N_int, self.qw_int_vec),
            (self.N_xi_int, self.N_xi_int, self.qw_int_vec),
        ]
        self.h_pot_q_SAB = SparseArrayBlocks(
            (self.parent.nu, self.parent.nq),
            (6, self.parent.kinematics.nq_node),
            h_pot_q_pairs,
        )
        self.K_sigma_SAB = SparseArrayBlocks(
            (self.parent.nu, self.parent.nu), (6, 6), h_pot_q_pairs
        )

    def sigma_qp_db(self, t, q, u):
        _eval = self.parent.kinematics._eval_internal_vec(self.N_int, self.N_xi_int, q)
        epsilon = np.hstack([_eval[1], _eval[2]]) / self.J_int_vec[:, None]
        sigma_db = self.material_model.sigma(
            epsilon, self.epsilon0_int, self.material_model_qp
        )

        sigma_qp = np.zeros((self.nquadrature_int_total, 6))
        sigma_qp[:, self.idx_db] = sigma_db[:, self.idx_db]
        return _eval, sigma_qp

    ############################
    # total energies and momenta
    ############################
    # the potential energies only work if
    # A) there is no coupling between the deformations
    # B) all of the coupled deformations are (not) db
    #    --> we put a warning when there are db and mx deformations
    def _E_pot_comp(self, t, q, la_c):
        if self._nDB > 0:
            msg = "E_pot_comp might not be correct if there are displacement-based deformations."
            warn(msg)
        _eval = self._eval_internal_vec(self.N_int, self.N_xi_int, q)
        epsilon_db = np.hstack([_eval[1], _eval[2]]) / self.J_int_vec[:, None]
        d_epsilon = epsilon_db - self.epsilon0_int

        la_sigma_nodes = np.zeros((self.nnodes_sigma, 6))
        la_sigma_nodes[:, self.idx_c] = la_c.reshape(self.nnodes_sigma, -1)
        la_sigma = self.Nc_int @ la_sigma_nodes

        C_qp = self.material_model.C_inv(self.material_model_qp)
        E_pot_i_star = 0.5 * np.einsum("ij,ijk,ik->i", la_sigma, C_qp, la_sigma)
        E_pot_i = np.sum(d_epsilon * la_sigma, axis=1) - E_pot_i_star
        E_pot = np.sum(E_pot_i * self.qw_int_vec * self.J_int_vec)
        return E_pot

    def E_pot(self, t, q):
        if self._nla_c > 0:
            warn("E_pot might not be correct if there are compliant deformations.")
        if self._nla_g > 0:
            warn("E_pot might not be correct if there are constrained deformations.")
        _eval = self.parent.kinematics._eval_internal_vec(self.N_int, self.N_xi_int, q)
        epsilon_db = np.hstack([_eval[1], _eval[2]]) / self.J_int_vec[:, None]
        epsilon = np.zeros_like(epsilon_db)
        epsilon[:, self.idx_db] = epsilon_db[:, self.idx_db]
        epsilon0 = np.zeros_like(epsilon_db)
        epsilon0[:, self.idx_db] = self.epsilon0_int[:, self.idx_db]
        E_pot_i = self.material_model.potential(
            epsilon, epsilon0, self.material_model_qp
        )
        E_pot = np.sum(E_pot_i * self.qw_int_vec * self.J_int_vec)
        return E_pot

    def E_pot_comp(self, t, q, la_c):
        if self._nDB > 0:
            msg = "E_pot_comp might not be correct if there are displacement-based deformations."
            warn(msg)
        _eval = self.parent.kinematics._eval_internal_vec(self.N_int, self.N_xi_int, q)
        epsilon_db = np.hstack([_eval[1], _eval[2]]) / self.J_int_vec[:, None]
        d_epsilon = epsilon_db - self.epsilon0_int

        la_sigma_nodes = np.zeros((self.nnodes_sigma, 6))
        la_sigma_nodes[:, self.idx_c] = la_c.reshape(self.nnodes_sigma, -1)
        la_sigma = self.Nc_int @ la_sigma_nodes

        C_qp = self.material_model.C_inv(self.material_model_qp)
        E_pot_i_star = 0.5 * np.einsum("ij,ijk,ik->i", la_sigma, C_qp, la_sigma)
        E_pot_i = np.sum(d_epsilon * la_sigma, axis=1) - E_pot_i_star
        E_pot = np.sum(E_pot_i * self.qw_int_vec * self.J_int_vec)
        return E_pot


class CosseratRod_internal_PG_IB(CosseratRod_Internal):
    def l_sigma(self, q):
        _, B_gamma_bar, B_kappa_bar = self.parent.kinematics._eval_internal_vec(
            self.N_int, self.N_xi_int, q
        )

        epsilon_bar = np.empty((self.nquadrature_int_total, 6))
        epsilon_bar[:, :3] = B_gamma_bar - self.B_gamma0_bar_int
        epsilon_bar[:, 3:] = B_kappa_bar - self.B_kappa0_bar_int

        l = self.Nc_int.T @ (epsilon_bar * self.qw_int_vec[:, None])

        l_c = l[:, self.idx_c].reshape(-1)
        l_g = l[:, self.idx_g].reshape(-1)
        return l_c, l_g

    def l_sigma_q(self, q):
        # compute l_sigma_q
        _eval, _deval = self.parent.kinematics._eval_internal_vec(
            self.N_int, self.N_xi_int, q, deval=True
        )
        A_IB = _eval[0]
        T, B_gamma_bar_P, B_kappa_bar_P = _deval

        # TODO: make sparse?
        # c_sigma_q_qp[N/N_xi, qpi, la_cDOF, qDOF]
        c_sigma_q_qp = np.empty(
            (2, self.nquadrature_int_total, 6, self.parent.kinematics.nq_node)
        )
        # to be multiplied with N_xi
        c_sigma_q_qp[1, :, :3, :3] = A_IB.transpose((0, 2, 1))
        c_sigma_q_qp[1, :, :3, 3:] = 0.0
        c_sigma_q_qp[1, :, 3:, :3] = 0.0
        c_sigma_q_qp[1, :, 3:, 3:] = T

        # to be multiplied with N
        c_sigma_q_qp[0, :, :, :3] = 0.0
        c_sigma_q_qp[0, :, :3, 3:] = B_gamma_bar_P
        c_sigma_q_qp[0, :, 3:, 3:] = B_kappa_bar_P

        return self.c_sigma_q_SAB.add_blocks(c_sigma_q_qp)

    def W_sigma(self, q):
        # compute W_sigma
        A_IB, B_gamma_bar, B_kappa_bar = self.parent.kinematics._eval_internal_vec(
            self.N_int, self.N_xi_int, q
        )

        gamma_bar_tilde = ax2skew(B_gamma_bar)
        kappa_bar_tilde = ax2skew(B_kappa_bar)

        # W_sigma_qp[N/N_xi, qpi, uDOF, la_cDOF]
        W_sigma_qp = np.empty((2, self.nquadrature_int_total, 6, 6))
        # to be multiplied with N_xi
        W_sigma_qp[1, :, :3, :3] = -A_IB
        W_sigma_qp[1, :, :3, 3:] = 0.0
        W_sigma_qp[1, :, 3:, :3] = 0.0
        W_sigma_qp[1, :, 3:, 3:] = -eye3

        # to be multiplied with N
        W_sigma_qp[0, :, :3, :] = 0.0
        W_sigma_qp[0, :, 3:, :3] = gamma_bar_tilde
        W_sigma_qp[0, :, 3:, 3:] = kappa_bar_tilde

        return self.W_sigma_SAB.add_blocks(W_sigma_qp)

    def Wla_sigma_q(self, q, la_c, la_g):
        la_sigma_nodes = np.zeros((self.nnodes_sigma, 6))
        if la_c is not None:
            la_sigma_nodes[:, self.idx_c] = la_c.reshape(self.nnodes_sigma, -1)
        if la_g is not None:
            la_sigma_nodes[:, self.idx_g] = la_g.reshape(self.nnodes_sigma, -1)
        sigma_qp = self.Nc_int @ la_sigma_nodes

        _eval, _deval = self.parent.kinematics._eval_internal_vec(
            self.N_int, self.N_xi_int, q, deval=True
        )
        A_IB = _eval[0]
        T, B_gamma_bar_P, B_kappa_bar_P = _deval
        I_n_P = np.einsum(
            "ijk,ikl->ijl",
            A_IB,
            np.cross(sigma_qp[:, :3, None], T, axisa=1, axisb=1, axisc=1),
        )

        # TODO: make sparse?
        # Wla_sigma_qp_qbar[N/N_xi, qpi, uDOF, qDOF]
        Wla_sigma_qp_qbar = np.zeros(
            (4, self.nquadrature_int_total, 6, self.parent.kinematics.nq_node)
        )
        # to be multiplied with N_xi <-> N
        Wla_sigma_qp_qbar[2, :, :3, 3:] = I_n_P

        # to be multiplied with N <-> N_xi
        Wla_sigma_qp_qbar[1, :, 3:, :3] = -np.cross(
            sigma_qp[:, :3, None], A_IB, axisa=1, axisb=2, axisc=1
        )  # A_IB.T in gamma -> axisb=2
        Wla_sigma_qp_qbar[1, :, 3:, 3:] = -np.cross(
            sigma_qp[:, 3:, None], T, axisa=1, axisb=1, axisc=1
        )

        # to be multiplied with N <-> N
        Wla_sigma_qp_qbar[0, :, 3:, 3:] = -(
            np.cross(sigma_qp[:, :3, None], B_gamma_bar_P, axisa=1, axisb=1, axisc=1)
            + np.cross(sigma_qp[:, 3:, None], B_kappa_bar_P, axisa=1, axisb=1, axisc=1)
        )
        return self.h_pot_q_SAB.add_blocks(Wla_sigma_qp_qbar)

    def K_sigma(self, q, la_c, la_g):
        la_sigma_nodes = np.zeros((self.nnodes_sigma, 6))
        if la_c is not None:
            la_sigma_nodes[:, self.idx_c] = la_c.reshape(self.nnodes_sigma, -1)
        if la_g is not None:
            la_sigma_nodes[:, self.idx_g] = la_g.reshape(self.nnodes_sigma, -1)
        sigma_qp = self.Nc_int @ la_sigma_nodes

        A_IB, B_gamma_bar, B_kappa_bar = self.parent.kinematics._eval_internal_vec(
            self.N_int, self.N_xi_int, q
        )

        # TODO: np.cross
        r_xi__phi = -np.einsum("ijk,ikl->ijl", A_IB, ax2skew(sigma_qp[:, :3]))
        phi_xi__phi = -0.5 * ax2skew(sigma_qp[:, 3:])

        phi__phi = np.einsum(
            "ijk,ikl->ijl", ax2skew(B_gamma_bar), ax2skew(sigma_qp[:, :3])
        ) + np.einsum("ijk,ikl->ijl", ax2skew(B_kappa_bar), ax2skew(sigma_qp[:, 3:]))
        phi__phi = 0.5 * (phi__phi + phi__phi.transpose(0, 2, 1))

        # TODO: make sparse?
        # K_qp[N/N_xi, qpi, uDOF, qDOF]
        K_qp = np.zeros((4, self.nquadrature_int_total, 6, 6))
        # to be multiplied with N_xi <-> N
        K_qp[2, :, :3, 3:] = r_xi__phi
        K_qp[2, :, 3:, 3:] = phi_xi__phi

        # to be multiplied with N <-> N_xi
        K_qp[1, :, 3:, :3] = r_xi__phi.transpose(0, 2, 1)
        K_qp[1, :, 3:, 3:] = -phi_xi__phi

        # to be multiplied with N <-> N
        K_qp[0, :, 3:, 3:] = phi__phi
        return self.K_sigma_SAB.add_blocks(K_qp)

    def assembler_callback(self):
        if self._nla_c > 0:
            self._c_la_c_coo()

    def _c_la_c_coo(self):
        C_qp = self.material_model.C_inv(self.material_model_qp)
        c_la_c = self.c_la_c_SAB.add_blocks(
            C_qp[None, :, self.idx_c[:, None], self.idx_c]
        )

        self.c_la_c = c_la_c
        self.c_la_c_inv = spsolve(c_la_c.tocsc(), eye_array(self._nla_c, format="csc"))

    def f_pot(self, t, q, u):
        _eval, sigma_qp = self.sigma_qp_db(t, q, u)

        # compute generalized internal forces based on evaluation and forces at quadrature points
        A_IB, B_gamma_bar, B_kappa_bar = _eval

        # h_pot_qp[N/N_xi, qpi, uDOF]
        h_pot_qp = np.empty((2, self.nquadrature_int_total, 6))
        # to be multiplied with N_xi
        h_pot_qp[1, :, :3] = -np.einsum("ijk,ik->ij", A_IB, sigma_qp[:, :3])
        h_pot_qp[1, :, 3:] = -sigma_qp[:, 3:]

        # to be multiplied with N
        h_pot_qp[0, :, :3] = 0.0
        h_pot_qp[0, :, 3:] = np.cross(B_gamma_bar, sigma_qp[:, :3]) + np.cross(
            B_kappa_bar, sigma_qp[:, 3:]
        )

        # add together and multiply with quadrature weights
        h_pot_nodes = self.N_int.T @ (
            h_pot_qp[0] * self.qw_int_vec[:, None]
        ) + self.N_xi_int.T @ (h_pot_qp[1] * self.qw_int_vec[:, None])
        return h_pot_nodes.reshape(-1)

    def f_pot_q(self, t, q, u):
        _eval, _deval = self.parent.kinematics._eval_internal_vec(
            self.N_int, self.N_xi_int, q, deval=True
        )
        epsilon = np.hstack([_eval[1], _eval[2]]) / self.J_int_vec[:, None]
        sigma_db = self.material_model.sigma(
            epsilon, self.epsilon0_int, self.material_model_qp
        )

        sigma_qp = np.zeros((self.nquadrature_int_total, 6))
        sigma_qp[:, self.idx_db] = sigma_db[:, self.idx_db]

        A_IB, B_gamma_bar, B_kappa_bar = _eval
        T, B_gamma_bar_P, B_kappa_bar_P = _deval

        ######################
        # material stiffness #
        ######################
        A_IB_transpose_to_J = A_IB.transpose(0, 2, 1) / self.J_int_vec[:, None, None]
        T_to_J = T / self.J_int_vec[:, None, None]

        sigma_epsilon = self.material_model.sigma_epsilon(
            epsilon, self.epsilon0_int, self.material_model_qp
        )
        B_n_gamma, B_n_kappa, B_m_gamma, B_m_kappa = sigma_epsilon

        # fmt: off
        B_n_P = (B_n_gamma @ B_gamma_bar_P + B_n_kappa @ B_kappa_bar_P) / self.J_int_vec[:, None, None]
        B_m_P = (B_m_gamma @ B_gamma_bar_P + B_m_kappa @ B_kappa_bar_P) / self.J_int_vec[:, None, None]
        # fmt: on

        # TODO: make sparse?
        # f_pot_qp_qbar[N/N_xi, qpi, uDOF, qDOF]
        f_pot_qp_qbar = np.zeros(
            (4, self.nquadrature_int_total, 6, self.parent.kinematics.nq_node)
        )
        # to be multiplied with N_xi <-> N_xi
        f_pot_qp_qbar[3, :, :3, :3] = -A_IB @ B_n_gamma @ A_IB_transpose_to_J
        f_pot_qp_qbar[3, :, :3, 3:] = -A_IB @ B_n_kappa @ T_to_J
        f_pot_qp_qbar[3, :, 3:, :3] = -B_m_gamma @ A_IB_transpose_to_J
        f_pot_qp_qbar[3, :, 3:, 3:] = -B_m_kappa @ T_to_J

        # to be multiplied with N_xi <-> N
        f_pot_qp_qbar[2, :, :3, 3:] = -A_IB @ B_n_P
        f_pot_qp_qbar[2, :, 3:, 3:] = -B_m_P

        # to be multiplied with N <-> N_xi
        f_pot_qp_qbar[1, :, 3:, :3] = (
            np.cross(B_gamma_bar[:, :, None], B_n_gamma, axisa=1, axisb=1, axisc=1)
            + np.cross(B_kappa_bar[:, :, None], B_m_gamma, axisa=1, axisb=1, axisc=1)
        ) @ A_IB_transpose_to_J
        f_pot_qp_qbar[1, :, 3:, 3:] = (
            np.cross(B_gamma_bar[:, :, None], B_n_kappa, axisa=1, axisb=1, axisc=1)
            + np.cross(B_kappa_bar[:, :, None], B_m_kappa, axisa=1, axisb=1, axisc=1)
        ) @ T_to_J

        # to be multiplied with N <-> N
        f_pot_qp_qbar[0, :, 3:, 3:] = np.cross(
            B_gamma_bar[:, :, None], B_n_P, axisa=1, axisb=1, axisc=1
        ) + np.cross(B_kappa_bar[:, :, None], B_m_P, axisa=1, axisb=1, axisc=1)

        ##################
        # geometric part #
        ##################
        # TODO: can we use this from Wla_sigma_q?
        I_n_P = np.einsum(
            "ijk,ikl->ijl",
            A_IB,
            np.cross(sigma_qp[:, :3, None], T, axisa=1, axisb=1, axisc=1),
        )

        # to be multiplied with N_xi <-> N
        f_pot_qp_qbar[2, :, :3, 3:] += I_n_P

        # to be multiplied with N <-> N_xi
        f_pot_qp_qbar[1, :, 3:, :3] -= np.cross(
            sigma_qp[:, :3, None], A_IB, axisa=1, axisb=2, axisc=1
        )  # A_IB.T in gamma -> axisb=2
        f_pot_qp_qbar[1, :, 3:, 3:] -= np.cross(
            sigma_qp[:, 3:, None], T, axisa=1, axisb=1, axisc=1
        )

        # to be multiplied with N <-> N
        f_pot_qp_qbar[0, :, 3:, 3:] -= np.cross(
            sigma_qp[:, :3, None], B_gamma_bar_P, axisa=1, axisb=1, axisc=1
        ) + np.cross(sigma_qp[:, 3:, None], B_kappa_bar_P, axisa=1, axisb=1, axisc=1)

        return self.h_pot_q_SAB.add_blocks(f_pot_qp_qbar)
