import numpy as np
from scipy.sparse import eye_array
from scipy.sparse.linalg import spsolve
from warnings import warn

from cardillo.math.algebra import ax2skew
from cardillo.math.approx_fprime import approx_fprime
from cardillo.utility.sparse_array_blocks import SparseArrayBlocks
from .cosseratRod_MatrixUtils import (
    bsr_add_block_diag,
    bsr_left,
    bsr_right,
    bsr_sym_skew,
)

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

        # eval functions
        self.parent.eval_stresses = self.eval_stresses
        self.parent.eval_strains = self.eval_strains

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

        # generalized force directions and their derivatives
        nu_node = self.parent.velocity.nu_node

        # W_sigma
        W_sigma_pairs = [
            (self.N_int, self.Nc_int, self.qw_int_vec),
            (self.N_xi_int, self.Nc_int, self.qw_int_vec),
        ]
        self.W_sigma_SAB = SparseArrayBlocks(
            (self.parent.nu, self.nla_sigma),
            (nu_node, 6),
            W_sigma_pairs,
            [(..., self.idx_c), (..., self.idx_g)],
        )

        # Wla_sigma_q
        # pairs [N/N_xi <-> N/N_xi] of the internal quadrature
        self.N_Nxi_pairs = [
            (self.N_int, self.N_int, self.qw_int_vec),
            (self.N_int, self.N_xi_int, self.qw_int_vec),
            (self.N_xi_int, self.N_int, self.qw_int_vec),
            (self.N_xi_int, self.N_xi_int, self.qw_int_vec),
        ]
        self.h_pot_q_SAB = SparseArrayBlocks(
            (self.parent.nu, self.parent.nq),
            (nu_node, self.parent.kinematics.nq_node),
            self.N_Nxi_pairs,
        )
        self.K_sigma_SAB = SparseArrayBlocks(
            (self.parent.nu, self.parent.nu), (nu_node, nu_node), self.N_Nxi_pairs
        )

    def sigma_qp_db(self, t, _eval, _eval_dot, deval=0):
        epsilon = np.hstack([_eval[1], _eval[2]]) / self.J_int_vec[:, None]
        epsilon_dot = np.hstack([_eval_dot[1], _eval_dot[2]]) / self.J_int_vec[:, None]
        sigma_db = self.material_model.sigma(
            epsilon, self.epsilon0_int, epsilon_dot, self.material_model_qp
        )

        sigma_qp = np.zeros((self.nquadrature_int_total, 6))
        sigma_qp[:, self.idx_db] = sigma_db[:, self.idx_db]
        if deval == 0:
            return sigma_qp

        # tangents as (nqp, 6, 6), only rows of displacement-based stresses
        sigma_epsilon_qp = self._tangent_db(
            self.material_model.sigma_epsilon(
                epsilon, self.epsilon0_int, epsilon_dot, self.material_model_qp
            )
        )
        sigma_epsilon_dot_qp = self._tangent_db(
            self.material_model.sigma_epsilon_dot(
                epsilon, self.epsilon0_int, epsilon_dot, self.material_model_qp
            )
        )
        return sigma_qp, sigma_epsilon_qp, sigma_epsilon_dot_qp

    def _tangent_db(self, blocks):
        B_n_gamma, B_n_kappa, B_m_gamma, B_m_kappa = blocks
        tangent = np.zeros((self.nquadrature_int_total, 6, 6))
        tangent[:, :3, :3] = B_n_gamma
        tangent[:, :3, 3:] = B_n_kappa
        tangent[:, 3:, :3] = B_m_gamma
        tangent[:, 3:, 3:] = B_m_kappa
        tangent_db = np.zeros_like(tangent)
        tangent_db[:, self.idx_db] = tangent[:, self.idx_db]
        return tangent_db

    ##########################################
    # linearization of displacement-based part
    ##########################################
    def DG_f_pot(self, t, q, u):
        # D + G = -f_pot_u, split into symmetric and skew-symmetric part
        return bsr_sym_skew(-self.f_pot_u(t, q, u))

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

    ############################
    # total energies and momenta
    ############################
    # the potential energies only work if
    # A) there is no coupling between the deformations
    # B) all of the coupled deformations are (not) db
    #    --> we put a warning when there are db and mx deformations
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

    # compliance length (dual to compliance force)
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
        # c_sigma_q_qp[N/N_xi, qpi, la_cDOF, qDOF] = epsilon_bar_qbar
        _, _deval = self.parent.kinematics._eval_internal_vec(
            self.N_int, self.N_xi_int, q, deval=1
        )
        epsilon_bar_qbar = _deval[0]
        return self.c_sigma_q_SAB.add_blocks(epsilon_bar_qbar)

    def l_sigma_dot_q(self, q, u):
        # TODO: do this without approx_fprime
        # TODO: this is in the fully constraint case None and somehow called for consistent initial conditions...
        l_c_dot_q = approx_fprime(q, lambda q_: self.W_sigma(q_)[0].T @ u)
        l_g_dot_q = approx_fprime(q, lambda q_: self.W_sigma(q_)[1].T @ u)
        return l_c_dot_q, l_g_dot_q

    def l_sigma_ddot(self, q, u, u_dot):
        # TODO: do this consistent, i.e., w/o q_dot
        W_c, W_g = self.W_sigma(q)
        l_c_dot_q, l_g_dot_q = self.l_sigma_dot_q(q, u)
        q_dot = self.parent.q_dot(0.0, q, u)

        l_c_ddot = W_c.T @ u_dot + l_c_dot_q @ q_dot
        l_g_ddot = W_g.T @ u_dot + l_g_dot_q @ q_dot
        return l_c_ddot, l_g_ddot

    ########################
    # evaluation functions #
    ########################
    def _eval_logic(self, n_per_element, n_ges):
        assert (n_per_element is not None) != (
            n_ges is not None
        ), "Either n_per_element or n_ges must be specified (not both)"

        if n_ges is not None:
            xis = np.linspace(0, 1, n_ges)
            els = self.parent.element_number(xis)

        else:
            xis = []
            els = []
            for el in range(self.parent.nelement):
                xi0, xi1 = self.parent.element_interval[el]
                xis.append(np.linspace(xi0, xi1, n_per_element))
                els.append(np.tile(el, n_per_element))
            xis = np.concatenate(xis)
            els = np.concatenate(els)

        return xis, els

    def eval_stresses(self, t, q, u, la_c, la_g, n_per_element=None, n_ges=None):
        xis, els = self._eval_logic(n_per_element, n_ges)

        # TODO: are there problems, if everything is DB?
        # stresses due to compliance and constraints
        Nc = self.parent.Nc(xis, els)
        la_sigma_nodes = np.zeros((self.nnodes_sigma, 6))
        if self._nla_c > 0:
            la_sigma_nodes[:, self.idx_c] = la_c[self.parent.la_cDOF].reshape(
                self.nnodes_sigma, -1
            )
        if self._nla_g > 0:
            # the internal constraints come first, cf. compose_g
            la_g_int = la_g[self.parent.la_gDOF][: self._nla_g]
            la_sigma_nodes[:, self.idx_g] = la_g_int.reshape(self.nnodes_sigma, -1)
        sigma = Nc @ la_sigma_nodes

        # stresses due to displacement based
        if self._nDB > 0:
            N, N_xi = self.parent.N(xis, els)
            # reference strains
            _, B_gamma0_bar, B_kappa0_bar = self.parent.kinematics._eval_internal_vec(
                N, N_xi, self.parent.Q
            )
            J = np.linalg.norm(B_gamma0_bar, axis=1)
            # current strains
            _, B_gamma_bar, B_kappa_bar = self.parent.kinematics._eval_internal_vec(
                N, N_xi, q[self.parent.qDOF]
            )

            epsilon = np.hstack([B_gamma_bar, B_kappa_bar]) / J[:, None]
            epsilon0 = np.hstack([B_gamma0_bar, B_kappa0_bar]) / J[:, None]
            prepare = self.material_model.prepare(xis)
            if u is None:
                epsilon_dot = np.zeros_like(epsilon)
            else:
                _eval_dot = self._eval_internal_vec_dot(
                    N, N_xi, q[self.parent.qDOF], u[self.parent.uDOF]
                )
                epsilon_dot = np.hstack(_eval_dot[1:]) / J[:, None]
            sigma_db = self.material_model.sigma(
                epsilon, epsilon0, epsilon_dot, prepare
            )
            sigma[:, self.idx_db] = sigma_db[:, self.idx_db]

        return xis, sigma[:, :3], sigma[:, 3:]

    def eval_strains(self, t, q, u, la_c, la_g, n_per_element=None, n_ges=None):
        xis, els = self._eval_logic(n_per_element, n_ges)

        epsilon = np.zeros((len(xis), 6))
        if self._nDB > 0:
            N, N_xi = self.parent.N(xis, els)
            # reference strains
            _, B_gamma0_bar, B_kappa0_bar = self.parent.kinematics._eval_internal_vec(
                N, N_xi, self.parent.Q
            )
            J = np.linalg.norm(B_gamma0_bar, axis=1)
            # current strains
            _, B_gamma_bar, B_kappa_bar = self.parent.kinematics._eval_internal_vec(
                N, N_xi, q[self.parent.qDOF]
            )

            epsilon[:, :3] = (B_gamma_bar - B_gamma0_bar) / J[:, None]
            epsilon[:, 3:] = (B_kappa_bar - B_kappa0_bar) / J[:, None]

        # strains from compliance
        if self._nla_c > 0:
            Nc = self.parent.Nc(xis, els)
            la_c_nodes = la_c[self.parent.la_cDOF].reshape(self.nnodes_sigma, -1)
            la_sigma = Nc @ la_c_nodes

            prepare = self.material_model.prepare(xis)
            C_inv = self.material_model.C_inv(prepare)
            C_inv_c = C_inv[:, self.idx_c[:, None], self.idx_c]
            epsilon[:, self.idx_c] = np.einsum("ijk,ik->ij", C_inv_c, la_sigma)

        # strains from constraints are always 0
        epsilon[:, self.idx_g] = 0.0

        return xis, epsilon[:, :3], epsilon[:, 3:]


class CosseratRod_internal_PG_IB(CosseratRod_Internal):
    def _geometric_stiffness_qp(self, sigma_qp, _deval):
        epsilon_bar_qbar, A_IB_qbar = _deval[:2]

        result = np.zeros(
            (4, self.nquadrature_int_total, 6, self.parent.kinematics.nq_node)
        )
        for i in range(2):
            # to be multiplied with N_xi <-> N/N_xi: -d(A_IB @ B_n)/dq
            result[2 + i, :, :3] = -np.einsum(
                "ijkl,ik->ijl", A_IB_qbar[i], sigma_qp[:, :3]
            )
            # to be multiplied with N <-> N/N_xi:
            # d(B_gamma_bar x B_n + B_kappa_bar x B_m)/dq
            result[i, :, 3:] = -np.cross(
                sigma_qp[:, :3, None],
                epsilon_bar_qbar[i, :, :3],
                axisa=1,
                axisb=1,
                axisc=1,
            ) - np.cross(
                sigma_qp[:, 3:, None],
                epsilon_bar_qbar[i, :, 3:],
                axisa=1,
                axisb=1,
                axisc=1,
            )
        return result

    def _eval_internal_vec_dot(self, N, N_xi, q, u, *, _eval=None):
        """Strain rates consistent with the virtual strains of the Petrov-Galerkin
        formulation, i.e., the virtual strain operator applied to u. Returns
        (B_Omega, B_gamma_bar_dot, B_kappa_bar_dot)."""
        if u @ u == 0:
            n = N.shape[0]
            zeros_n3 = np.zeros((n, 3))
            return (zeros_n3, zeros_n3, zeros_n3)

        if _eval is None:
            _eval = self.parent.kinematics._eval_internal_vec(N, N_xi, q)

        unodes = u.reshape(self.parent.nnodes, -1)
        B_Omega = N @ unodes[:, 3:]
        vO_xi = N_xi @ unodes

        B_gamma_bar_dot = np.einsum("ijk,ij->ik", _eval[0], vO_xi[:, :3]) - np.cross(
            B_Omega, _eval[1]
        )
        B_kappa_bar_dot = vO_xi[:, 3:] - np.cross(B_Omega, _eval[2])

        _eval_dot = (B_Omega, B_gamma_bar_dot, B_kappa_bar_dot)
        return _eval_dot

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
            self.N_int, self.N_xi_int, q, deval=1
        )
        Wla_sigma_qp_qbar = self._geometric_stiffness_qp(sigma_qp, _deval)
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
        # TODO: keep checking of the sign. for the moment, no minus is correct, as shown by the updated Helix experiment
        r_xi__phi = np.einsum("ijk,ikl->ijl", A_IB, ax2skew(sigma_qp[:, :3]))
        phi_xi__phi = 0.5 * ax2skew(sigma_qp[:, 3:])

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

    def f_pot(self, t, q, u):
        _eval = self.parent.kinematics._eval_internal_vec(self.N_int, self.N_xi_int, q)
        _eval_dot = self._eval_internal_vec_dot(
            self.N_int, self.N_xi_int, q, u, _eval=_eval
        )
        sigma_qp = self.sigma_qp_db(t, _eval, _eval_dot)

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
            self.N_int, self.N_xi_int, q, deval=1
        )
        _eval_dot = self._eval_internal_vec_dot(
            self.N_int, self.N_xi_int, q, u, _eval=_eval
        )
        sigma_qp, C_qp, D_qp = self.sigma_qp_db(t, _eval, _eval_dot, deval=1)

        A_IB, B_gamma_bar, B_kappa_bar = _eval
        epsilon_bar_qbar, A_IB_qbar = _deval[:2]
        B_Omega = _eval_dot[0]

        ######################################
        # material stiffness (incl. viscous) #
        ######################################
        # B_gamma_bar_dot = A_IB.T @ I_v_xi - B_Omega x B_gamma_bar
        # B_kappa_bar_dot = B_Omega_xi - B_Omega x B_kappa_bar
        epsilon_bar_dot_qbar = np.empty_like(epsilon_bar_qbar)
        epsilon_bar_dot_qbar[..., :3, :] = -np.cross(
            B_Omega[:, :, None],
            epsilon_bar_qbar[..., :3, :],
            axisa=1,
            axisb=-2,
            axisc=-2,
        )
        epsilon_bar_dot_qbar[..., 3:, :] = -np.cross(
            B_Omega[:, :, None],
            epsilon_bar_qbar[..., 3:, :],
            axisa=1,
            axisb=-2,
            axisc=-2,
        )
        # derivative of A_IB.T @ I_v_xi
        I_v_xi = self.N_xi_int @ u.reshape(self.parent.nnodes, -1)[:, :3]
        epsilon_bar_dot_qbar[..., :3, :] += np.einsum(
            "aijkl,ij->aikl", A_IB_qbar, I_v_xi
        )

        sigma_qbar = (
            C_qp @ epsilon_bar_qbar + D_qp @ epsilon_bar_dot_qbar
        ) / self.J_int_vec[:, None, None]

        # f_pot_qp_qbar[N/N_xi <-> N/N_xi, qpi, uDOF, qDOF]
        f_pot_qp_qbar = self._geometric_stiffness_qp(sigma_qp, _deval)
        for i in range(2):
            f_pot_qp_qbar[2 + i] += self._B_xi_T_minus(A_IB, sigma_qbar[i])
            f_pot_qp_qbar[i] += self._B_0_T_minus(
                B_gamma_bar, B_kappa_bar, sigma_qbar[i]
            )

        return self.h_pot_q_SAB.add_blocks(f_pot_qp_qbar)

    def f_pot_u(self, t, q, u):
        _eval = self.parent.kinematics._eval_internal_vec(self.N_int, self.N_xi_int, q)
        _eval_dot = self._eval_internal_vec_dot(
            self.N_int, self.N_xi_int, q, u, _eval=_eval
        )
        _, _, D_qp = self.sigma_qp_db(t, _eval, _eval_dot, deval=1)

        A_IB, B_gamma_bar, B_kappa_bar = _eval

        # epsilon_bar_dot_ubar[N/N_xi, qpi, epsilonDOF, uDOF]
        epsilon_bar_dot_ubar = np.zeros((2, self.nquadrature_int_total, 6, 6))
        # to be multiplied with N_xi
        epsilon_bar_dot_ubar[1, :, :3, :3] = A_IB.transpose(0, 2, 1)
        epsilon_bar_dot_ubar[1, :, 3:, 3:] = eye3
        # to be multiplied with N
        epsilon_bar_dot_ubar[0, :, :3, 3:] = np.cross(
            B_gamma_bar[:, :, None], eye3[None], axisa=1, axisb=1, axisc=1
        )
        epsilon_bar_dot_ubar[0, :, 3:, 3:] = np.cross(
            B_kappa_bar[:, :, None], eye3[None], axisa=1, axisb=1, axisc=1
        )

        sigma_ubar = D_qp @ epsilon_bar_dot_ubar / self.J_int_vec[:, None, None]

        # f_pot_qp_ubar[N/N_xi, qpi, uDOF, uDOF]
        f_pot_qp_ubar = np.empty((4, self.nquadrature_int_total, 6, 6))
        # to be multiplied with N_xi <-> N_xi, N_xi <-> N, N <-> N_xi, N <-> N
        f_pot_qp_ubar[3] = self._B_xi_T_minus(A_IB, sigma_ubar[1])
        f_pot_qp_ubar[2] = self._B_xi_T_minus(A_IB, sigma_ubar[0])
        f_pot_qp_ubar[1] = self._B_0_T_minus(B_gamma_bar, B_kappa_bar, sigma_ubar[1])
        f_pot_qp_ubar[0] = self._B_0_T_minus(B_gamma_bar, B_kappa_bar, sigma_ubar[0])

        return self.K_sigma_SAB.add_blocks(f_pot_qp_ubar)

    def KN_f_pot(self, t, q, u):
        # TODO: stiffness w.r.t. the nodal (Lie group) increments, cf. K_sigma
        warn("KN_f_pot not implemented for PG")
        return None, None

    # f_pot = N_xi.T @ (-B_xi.T @ sigma) + N.T @ (-B_0.T @ sigma), where
    # B_xi = diag(A_IB.T, eye3) and B_0 = [[0, skew(gamma_bar)], [0, skew(kappa_bar)]]
    @staticmethod
    def _B_xi_T_minus(A_IB, X):
        result = np.empty_like(X)
        result[:, :3] = -A_IB @ X[:, :3]
        result[:, 3:] = -X[:, 3:]
        return result

    @staticmethod
    def _B_0_T_minus(B_gamma_bar, B_kappa_bar, X):
        result = np.zeros_like(X)
        result[:, 3:] = np.cross(
            B_gamma_bar[:, :, None], X[:, :3], axisa=1, axisb=1, axisc=1
        ) + np.cross(B_kappa_bar[:, :, None], X[:, 3:], axisa=1, axisb=1, axisc=1)
        return result


class CosseratRod_internal_BG(CosseratRod_Internal):
    """Bubnov-Galerkin formulation, i.e., u = q_dot and the virtual work is
    tested with delta q. The generalized forces are
    f_BG(q, q_dot) = -int epsilon_bar_q.T @ sigma(epsilon, epsilon_dot)."""

    def set_reference_strains(self, Q):
        super().set_reference_strains(Q)

        # f_BG lives in q-space, i.e., blocks (nq_node, nq_node), which differ from
        # the blocks (nu_node, nq_node) of h_pot_q_SAB for DBG
        nq_node = self.parent.kinematics.nq_node
        self.f_BG_q_SAB = SparseArrayBlocks(
            (self.parent.nq, self.parent.nq), (nq_node, nq_node), self.N_Nxi_pairs
        )

    def _q_dot_qbar(self, N, N_xi, q, u):
        """q_dot_qbar[N/N_xi, qpi, qDOF], i.e., interpolated q_dot = u."""
        u_nodes = u.reshape(self.parent.nnodes, -1)
        return np.array([N @ u_nodes, N_xi @ u_nodes])

    def _eval_internal_vec_dot(self, N, N_xi, q, u, *, _deval=None):
        """Strain rates epsilon_bar_dot = epsilon_bar_q @ q_dot with q_dot = q_dot(q, u).
        Returns (None, B_gamma_bar_dot, B_kappa_bar_dot)."""
        if _deval is None:
            _, _deval = self.parent.kinematics._eval_internal_vec(N, N_xi, q, deval=1)
        q_dot_qbar = self._q_dot_qbar(N, N_xi, q, u)
        epsilon_bar_dot = np.einsum("apik,apk->pi", _deval[0], q_dot_qbar)
        return (None, epsilon_bar_dot[:, :3], epsilon_bar_dot[:, 3:])

    def _f_BG(self, sigma_qp, epsilon_bar_qbar):
        """f_BG = -int epsilon_bar_q.T @ sigma"""
        f_BG_qp = -np.einsum("apik,pi->apk", epsilon_bar_qbar, sigma_qp)
        f_BG_nodes = self.N_int.T @ (
            f_BG_qp[0] * self.qw_int_vec[:, None]
        ) + self.N_xi_int.T @ (f_BG_qp[1] * self.qw_int_vec[:, None])
        return f_BG_nodes.reshape(-1)

    def _f_BG_q_dot(self, D_qp, epsilon_bar_qbar):
        """Derivative of f_BG w.r.t. q_dot."""
        D_epsilon_bar_qbar = D_qp @ epsilon_bar_qbar / self.J_int_vec[:, None, None]
        f_BG_q_dot_qp = -np.einsum(
            "apik,bpil->abpkl", epsilon_bar_qbar, D_epsilon_bar_qbar
        )
        return self.f_BG_q_SAB.add_blocks(
            f_BG_q_dot_qp.reshape(4, *f_BG_q_dot_qp.shape[2:])
        )

    def _f_BG_q_geometric(self, sigma_qp, epsilon_bar_qbarqbar):
        """Derivative of f_BG w.r.t. q for fixed sigma, as qp_contributions."""
        return -np.einsum("pi,abpikl->abpkl", sigma_qp, epsilon_bar_qbarqbar)

    def _f_BG_and_derivatives(self, t, q, u):
        """f_BG, its derivative w.r.t. q for fixed q_dot and w.r.t. q_dot."""
        N, N_xi = self.N_int, self.N_xi_int
        _eval, _deval, _ddeval = self.parent.kinematics._eval_internal_vec(
            N, N_xi, q, deval=2
        )
        _eval_dot = self._eval_internal_vec_dot(N, N_xi, q, u, _deval=_deval)
        q_dot_qbar = self._q_dot_qbar(N, N_xi, q, u)
        sigma_qp, C_qp, D_qp = self.sigma_qp_db(t, _eval, _eval_dot, deval=1)
        epsilon_bar_qbar = _deval[0]
        epsilon_bar_qbarqbar = _ddeval[0]

        # derivative of epsilon_bar_dot w.r.t. q for fixed q_dot
        epsilon_bar_dot_qbar = np.einsum(
            "abpijk,apj->bpik", epsilon_bar_qbarqbar, q_dot_qbar
        )
        sigma_qbar = (
            C_qp @ epsilon_bar_qbar + D_qp @ epsilon_bar_dot_qbar
        ) / self.J_int_vec[:, None, None]

        f_BG_q_qp = -np.einsum(
            "apik,bpil->abpkl", epsilon_bar_qbar, sigma_qbar
        ) + self._f_BG_q_geometric(sigma_qp, epsilon_bar_qbarqbar)
        f_BG_q = self.f_BG_q_SAB.add_blocks(f_BG_q_qp.reshape(4, *f_BG_q_qp.shape[2:]))

        f_BG = self._f_BG(sigma_qp, epsilon_bar_qbar)
        f_BG_q_dot = self._f_BG_q_dot(D_qp, epsilon_bar_qbar)
        return f_BG, f_BG_q, f_BG_q_dot

    def _sigma_qp_la(self, la_c, la_g):
        la_sigma_nodes = np.zeros((self.nnodes_sigma, 6))
        if la_c is not None:
            la_sigma_nodes[:, self.idx_c] = la_c.reshape(self.nnodes_sigma, -1)
        if la_g is not None:
            la_sigma_nodes[:, self.idx_g] = la_g.reshape(self.nnodes_sigma, -1)
        return self.Nc_int @ la_sigma_nodes

    def _Wla_BG_and_derivative(self, q, la_c, la_g):
        """W_BG @ la = f_BG(sigma(la)) and its derivative w.r.t. q."""
        sigma_qp = self._sigma_qp_la(la_c, la_g)
        _, _deval, _ddeval = self.parent.kinematics._eval_internal_vec(
            self.N_int, self.N_xi_int, q, deval=2
        )
        f_BG_q_qp = self._f_BG_q_geometric(sigma_qp, _ddeval[0])
        f_BG_q = self.f_BG_q_SAB.add_blocks(f_BG_q_qp.reshape(4, *f_BG_q_qp.shape[2:]))
        return self._f_BG(sigma_qp, _deval[0]), f_BG_q

    ###########################
    # displacement-based part #
    ###########################
    def f_pot(self, t, q, u):
        N, N_xi = self.N_int, self.N_xi_int
        _eval, _deval = self.parent.kinematics._eval_internal_vec(N, N_xi, q, deval=1)
        _eval_dot = self._eval_internal_vec_dot(N, N_xi, q, u, _deval=_deval)
        sigma_qp = self.sigma_qp_db(t, _eval, _eval_dot)
        return self._f_BG(sigma_qp, _deval[0])

    def f_pot_q(self, t, q, u):
        # q_dot = u
        _, f_BG_q, _ = self._f_BG_and_derivatives(t, q, u)
        return f_BG_q

    def f_pot_u(self, t, q, u):
        # q_dot = u
        N, N_xi = self.N_int, self.N_xi_int
        _eval, _deval = self.parent.kinematics._eval_internal_vec(N, N_xi, q, deval=1)
        _eval_dot = self._eval_internal_vec_dot(N, N_xi, q, u, _deval=_deval)
        _, _, D_qp = self.sigma_qp_db(t, _eval, _eval_dot, deval=1)
        return self._f_BG_q_dot(D_qp, _deval[0])

    def KN_f_pot(self, t, q, u):
        # u = q_dot: K + N = -f_pot_q, split into symmetric and skew-symmetric part
        return bsr_sym_skew(-self.f_pot_q(t, q, u))

    ######################################
    # compliance and constraint stresses #
    ######################################
    def W_sigma(self, q):
        # W_BG = -l_sigma_q.T
        return tuple(None if l_q is None else -l_q.T for l_q in self.l_sigma_q(q))

    def Wla_sigma_q(self, q, la_c, la_g):
        return self._Wla_BG_and_derivative(q, la_c, la_g)[1]

    def K_sigma(self, q, la_c, la_g):
        # u = q_dot
        return self.Wla_sigma_q(q, la_c, la_g)


class CosseratRod_internal_DBG(CosseratRod_internal_BG):
    """Bubnov-Galerkin formulation with discrete nullspace projection, i.e.,
    the generalized forces of the Bubnov-Galerkin formulation f_BG(q, q_dot)
    are projected with the nodal kinematic equation q_dot = B(q) @ u, i.e.,
    f = B(q).T @ f_BG(q, B(q) @ u).

    As B is block diagonal w.r.t. the nodes, all projections are done on the
    blocks of the bsr arrays, which keeps their sparsity pattern."""

    def _q_dot_qbar(self, N, N_xi, q, u):
        """q_dot_qbar[N/N_xi, qpi, qDOF], i.e., interpolated q_dot = B(q) @ u."""
        u_nodes = u.reshape(self.parent.nnodes, -1)
        B = self.parent.kin_eq.B_blocks(q)
        q_dot_nodes = np.einsum("nij,nj->ni", B, u_nodes)
        return np.array([N @ q_dot_nodes, N_xi @ q_dot_nodes])

    ###########################
    # displacement-based part #
    ###########################
    def f_pot(self, t, q, u):
        return self.parent.kin_eq.BT_f(q, super().f_pot(t, q, u))

    def f_pot_q(self, t, q, u):
        f_BG, f_BG_q, f_BG_q_dot = self._f_BG_and_derivatives(t, q, u)

        # q_dot = B(q) @ u, same pattern for f_BG_q and f_BG_q_dot
        f_BG_q.data += bsr_right(f_BG_q_dot, self.parent.kin_eq.Bu_q_blocks(u)).data

        B_T = self.parent.kin_eq.B_blocks(q).transpose(0, 2, 1)
        return bsr_add_block_diag(
            bsr_left(B_T, f_BG_q), self.parent.kin_eq.BT_f_q_blocks(q, f_BG)
        )

    def f_pot_u(self, t, q, u):
        B = self.parent.kin_eq.B_blocks(q)
        return bsr_left(B.transpose(0, 2, 1), bsr_right(super().f_pot_u(t, q, u), B))

    def KN_f_pot(self, t, q, u):
        # K + N = -f_pot_q @ B, split into symmetric and skew-symmetric part
        f_pot_q_B = bsr_right(self.f_pot_q(t, q, u), self.parent.kin_eq.B_blocks(q))
        return bsr_sym_skew(-f_pot_q_B)

    ######################################
    # compliance and constraint stresses #
    ######################################
    def W_sigma(self, q):
        B_T = self.parent.kin_eq.B_blocks(q).transpose(0, 2, 1)
        nq_node = self.parent.kinematics.nq_node
        W = []
        for l_q in self.l_sigma_q(q):
            if l_q is None:
                W.append(None)
                continue
            # W_BG = -l_q.T in bsr format with blocks (nq_node, nla_node)
            W_BG = -l_q.T.tobsr(blocksize=(nq_node, l_q.blocksize[0]))
            W.append(bsr_left(B_T, W_BG))
        return tuple(W)

    def Wla_sigma_q(self, q, la_c, la_g):
        f_BG, f_BG_q = self._Wla_BG_and_derivative(q, la_c, la_g)
        B_T = self.parent.kin_eq.B_blocks(q).transpose(0, 2, 1)
        return bsr_add_block_diag(
            bsr_left(B_T, f_BG_q), self.parent.kin_eq.BT_f_q_blocks(q, f_BG)
        )

    def K_sigma(self, q, la_c, la_g):
        # derivative w.r.t. the nodal velocity-like increments, i.e., q_dot = B @ u
        return bsr_right(
            self.Wla_sigma_q(q, la_c, la_g), self.parent.kin_eq.B_blocks(q)
        )
