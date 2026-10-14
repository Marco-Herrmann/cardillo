from abc import ABC
import numpy as np
from scipy.sparse import block_diag, csr_array

from cardillo.math.rotations import (
    Exp_SO3_quat,
    Log_SO3_quat,
    T_SO3_inv_quat,
    T_SO3_inv_quat_P,
    Exp_SO3_R9,
    Log_SO3_R9,
    T_SO3_inv_R9,
    T_SO3_inv_R9_R9,
    quatprod,
    axis_angle2quat,
    Exp_SO3,
    SE3,
    Exp_SE3,
)
from cardillo.utility.coo_matrix import CooMatrix
from .cosseratRod_MatrixUtils import bsr_block_diag

zeros3 = np.zeros(3, dtype=float)
eye3 = np.eye(3, dtype=float)


class CosseratRod_kin_constraints(ABC):
    def __init__(self, parent, parametrization, projection):
        self.parent = parent
        self.nnodes = self.parent.nnodes

        assert parametrization in ["Quaternion", "R12", "SE3"]

        if parametrization in ["Quaternion", "SE3"]:
            self.nq_node = 7
            self.nla_g = self.nnodes

            self._Exp_SO3 = Exp_SO3_quat
            self._Log_SO3 = Log_SO3_quat

            # rows/cols for g_q
            # TODO: check this
            self._g_S_q_row = np.repeat(np.arange(self.nnodes), 4)
            self._g_S_q_col = (
                (3 + 7 * np.arange(self.nnodes))[:, None] + np.arange(4)
            ).ravel()

            self.g = self.g_quat
            self.g_q = self.g_q_quat

        else:
            self.nq_node = 12
            self.nla_g = self.nnodes * 6

            self._Exp_SO3 = Exp_SO3_R9
            self._Log_SO3 = Log_SO3_R9

            # rows/cols for g_q
            rows = []
            cols = []

            for inode in range(self.nnodes):
                DOF0 = 12 * inode
                d1DOF = DOF0 + np.arange(3) + 3
                d2DOF = DOF0 + np.arange(3) + 6
                d3DOF = DOF0 + np.arange(3) + 9
                # g0 = d1@ d1 - 1
                rows.extend([6 * inode + 0] * 3)
                cols.extend(d1DOF)

                # g1 = d2@d2 - 1
                rows.extend([6 * inode + 1] * 3)
                cols.extend(d2DOF)

                # g2 = d3@d3 - 1
                rows.extend([6 * inode + 2] * 3)
                cols.extend(d3DOF)

                # g3 = d1@d2
                rows.extend([6 * inode + 3] * 6)
                cols.extend([*d1DOF, *d2DOF])

                # g4 = d2@d3
                rows.extend([6 * inode + 4] * 6)
                cols.extend([*d2DOF, *d3DOF])

                # g5 = d3@d1
                rows.extend([6 * inode + 5] * 6)
                cols.extend([*d1DOF, *d3DOF])

            self._g_S_q_row = np.asarray(rows)
            self._g_S_q_col = np.asarray(cols)

            self.g = self.g_R9
            self.g_q = self.g_q_R9

        self._nla_g = self.nnodes * (self.nq_node - 6)
        if projection == False:
            self.include_g = True
            self.include_g_S = False
        else:
            self.include_g = False
            self.include_g_S = True

    def g_quat(self, t, q):
        qnodes = q.reshape(self.nnodes, -1)
        return np.sum(qnodes[:, 3:] ** 2, axis=1) - 1

    def g_q_quat(self, t, q):
        qnodes = q.reshape(self.nnodes, -1)
        coo = CooMatrix((self.nla_g, self.parent.nq))
        coo.data = 2 * qnodes[:, 3:].reshape(-1)
        coo.row = self._g_S_q_row
        coo.col = self._g_S_q_col
        return coo

    def g_R9(self, t, q):
        qnodes = q.reshape(self.nnodes, -1)
        d1 = qnodes[:, 3:6]
        d2 = qnodes[:, 6:9]
        d3 = qnodes[:, 9:12]

        gnodes = np.column_stack(
            [
                np.sum(d1 * d1, axis=1) - 1.0,
                np.sum(d2 * d2, axis=1) - 1.0,
                np.sum(d3 * d3, axis=1) - 1.0,
                np.sum(d1 * d2, axis=1),
                np.sum(d2 * d3, axis=1),
                np.sum(d3 * d1, axis=1),
            ]
        )

        g = gnodes.reshape(-1)
        return g

    def g_q_R9(self, t, q):
        qnodes = q.reshape(self.nnodes, -1)
        d1 = qnodes[:, 3:6]
        d2 = qnodes[:, 6:9]
        d3 = qnodes[:, 9:12]
        data = np.concatenate(
            [
                2 * d1,
                2 * d2,
                2 * d3,
                np.concatenate([d2, d1], axis=1),
                np.concatenate([d3, d2], axis=1),
                np.concatenate([d3, d1], axis=1),
            ],
            axis=1,
        )

        coo = CooMatrix((self.nla_g, self.parent.nq))
        coo.data = data.ravel()

        coo.row = self._g_S_q_row
        coo.col = self._g_S_q_col
        return coo

    def W_g(self, t, q):
        return self.g_q(t, q).T

    def Wla_g_q(self, t, q, la_g):
        # g is quadratic in q, hence W_g @ la_g is linear in q and its
        # derivative is independent of q
        blocks = np.zeros((self.nnodes, self.nq_node, self.nq_node))
        if self.nq_node == 7:
            # g = P @ P - 1
            blocks[:, 3:, 3:] = 2.0 * la_g[:, None, None] * np.eye(4)
        else:
            # g = (d1 @ d1 - 1, d2 @ d2 - 1, d3 @ d3 - 1, d1 @ d2, d2 @ d3, d3 @ d1)
            la = la_g.reshape(self.nnodes, 6)
            S = np.empty((self.nnodes, 3, 3))
            S[:, [0, 1, 2], [0, 1, 2]] = 2.0 * la[:, :3]
            S[:, 0, 1] = S[:, 1, 0] = la[:, 3]
            S[:, 1, 2] = S[:, 2, 1] = la[:, 4]
            S[:, 2, 0] = S[:, 0, 2] = la[:, 5]
            blocks[:, 3:, 3:] = np.einsum("nij,kl->nikjl", S, eye3).reshape(
                self.nnodes, 9, 9
            )
        # bsr keeps the zeros, i.e., the sparsity pattern is independent of la_g
        return bsr_block_diag(blocks)

    def g_dot(self, t, q, u): ...
    def g_dot_q(self, t, q, u): ...
    def g_dot_u(self, t, q): ...
    def g_ddot(self, t, q, u, u_dot): ...


class CosseratRod_kin_trivial(CosseratRod_kin_constraints):
    def __init__(self, parent, parametrization, projection):
        super().__init__(parent, parametrization, projection)
        self.parent.q_dot = self.q_dot
        self.parent.q_dot_u = self.q_dot_u

        # TODO: sparse diag?
        self.eye_nu = np.eye(self.parent.nu)

    def q_dot(self, t, q, u):
        return u

    def q_dot_u(self, t, q):
        return self.eye_nu

    # g is quadratic in q, hence g_q is linear in q and with q_dot = u:
    # g_dot = g_q(q) @ u, g_dot_q = g_q(u), g_ddot = g_q(q) @ u_dot + g_q(u) @ u
    def g_dot(self, t, q, u):
        return self.g_q(t, q).tocsr() @ u

    def g_dot_q(self, t, q, u):
        return self.g_q(t, u)

    def g_dot_u(self, t, q):
        return self.g_q(t, q)

    def g_ddot(self, t, q, u, u_dot):
        return self.g_q(t, q).tocsr() @ u_dot + self.g_q(t, u).tocsr() @ u

    def KN_g(self, t, q, la_g):
        # u = q_dot
        return self.Wla_g_q(t, q, la_g), None


class CosseratRod_rP_dot_from_vO_IB(CosseratRod_kin_constraints):
    def __init__(self, parent, parametrization, projection):
        super().__init__(parent, parametrization, projection)
        self.parent.q_dot = self.q_dot
        self.parent.q_dot_u = self.q_dot_u
        self.parent.q_dot_q = self.q_dot_q
        self.parent.Lie_update = self.Lie_update

        if parametrization in ["Quaternion", "SE3"]:
            self._T_IB_inv = T_SO3_inv_quat
            self._T_IB_inv_P = T_SO3_inv_quat_P(None)  # evaluate as it is constant
        else:
            self._T_IB_inv = T_SO3_inv_R9
            self._T_IB_inv_P = T_SO3_inv_R9_R9(None)  # evaluate as it is constant

    def q_dot(self, t, q, u):
        qnodes = q.reshape(self.nnodes, self.nq_node)
        unodes = u.reshape(self.nnodes, 6)

        qnodes_dot = np.empty((self.nnodes, self.nq_node), dtype=np.common_type(q, u))
        qnodes_dot[:, :3] = unodes[:, :3]
        qnodes_dot[:, 3:] = np.einsum(
            "ijk,ik->ij", self._T_IB_inv(qnodes[:, 3:]), unodes[:, 3:]
        )

        return qnodes_dot.reshape(-1)

    def q_dot_q(self, t, q, u):
        return csr_array(block_diag(self.Bu_q_blocks(u)))

    def q_dot_u(self, t, q):
        return csr_array(block_diag(self.B_blocks(q)))  # this keeps the 0's

    ##########################################
    # nodal blocks of the kinematic equation #
    ##########################################
    # q_dot = B(q) @ u with block diagonal B(q), where each node has the block
    # [[eye3, 0], [0, T_IB_inv(P)]] (nq_node, 6). The functions below work for
    # an arbitrary number of nodes, e.g., the nodes of an element.
    def B_blocks(self, q):
        """Nodal blocks (nq_node, 6) of B(q)."""
        q_nodes = q.reshape(-1, self.nq_node)
        B = np.zeros((len(q_nodes), self.nq_node, 6), dtype=q.dtype)
        B[:, :3, :3] = eye3
        B[:, 3:, 3:] = self._T_IB_inv(q_nodes[:, 3:])
        return B

    def B_q_blocks(self, q):
        """Nodal blocks (nq_node, 6, nq_node) of the derivative of B(q)."""
        nnodes = len(q) // self.nq_node
        B_q = np.zeros((nnodes, self.nq_node, 6, self.nq_node))
        B_q[:, 3:, 3:, 3:] = self._T_IB_inv_P
        return B_q

    def Bu_q_blocks(self, u):
        """Nodal blocks (nq_node, nq_node) of d(B(q) @ u)/dq."""
        u_nodes = u.reshape(-1, 6)
        blocks = np.zeros((len(u_nodes), self.nq_node, self.nq_node))
        blocks[:, 3:, 3:] = np.einsum("jkl,ik->ijl", self._T_IB_inv_P, u_nodes[:, 3:])
        return blocks

    def BT_f(self, q, f):
        """B(q).T @ f"""
        f_nodes = f.reshape(-1, self.nq_node)
        return np.einsum("nqk,nq->nk", self.B_blocks(q), f_nodes).reshape(-1)

    def BT_f_q_blocks(self, q, f):
        """Nodal blocks (6, nq_node) of d(B(q).T @ f)/dq for fixed f."""
        f_nodes = f.reshape(-1, self.nq_node)
        blocks = np.zeros((len(f_nodes), 6, self.nq_node))
        blocks[:, 3:, 3:] = np.einsum("jkl,ij->ikl", self._T_IB_inv_P, f_nodes[:, 3:])
        return blocks

    def Lie_update(self, t, q, Delta_s):
        # return self.Lie_update_R3xSO3_B(t, q, Delta_s)
        # return self.Lie_update_R3xSO3_I(t, q, Delta_s)
        return self.Lie_update_SE3_B(t, q, Delta_s)
        # return self.Lie_update_SE3_I(t, q, Delta_s)

    def Lie_update_R3xSO3_B(self, t, q, Delta_s):
        q_nodes = q.reshape(self.nnodes, self.nq_node)
        Delta_s_nodes = Delta_s.reshape(self.nnodes, 6)

        # positional update
        q_nodes[:, :3] += Delta_s_nodes[:, :3]

        # rotational update
        # TODO: figure out direct relation
        # Delta_phi = Delta_s_nodes[:, 3:]
        # # TODO: avoid division by 0, make new function: psi_to_quat
        # quat_rel = axis_angle2quat(Delta_phi_i / Delta_phi, Delta_phi)
        quat_rel = Log_SO3_quat(Exp_SO3(Delta_s_nodes[:, 3:]))

        q_nodes[:, 3:] = quatprod(q_nodes[:, 3:], quat_rel)
        return q_nodes.reshape(-1)

    def Lie_update_R3xSO3_I(self, t, q, Delta_s):
        q_nodes = q.reshape(self.nnodes, self.nq_node)
        Delta_s_nodes = Delta_s.reshape(self.nnodes, 6)

        # positional update
        q_nodes[:, :3] += Delta_s_nodes[:, :3]

        # bring rotation update to I-system
        A_IB = self._Exp_SO3(q_nodes[:, 3:])
        Delta_s_nodes[:, 3:] = np.einsum("ijk,ik->ij", A_IB, Delta_s_nodes[:, 3:])

        # rotational update
        # TODO: figure out direct relation
        # Delta_phi = Delta_s_nodes[:, 3:]
        # # TODO: avoid division by 0, make new function: psi_to_quat
        # quat_rel = axis_angle2quat(Delta_phi_i / Delta_phi, Delta_phi)
        quat_rel = Log_SO3_quat(Exp_SO3(Delta_s_nodes[:, 3:]))

        q_nodes[:, 3:] = quatprod(quat_rel, q_nodes[:, 3:])
        return q_nodes.reshape(-1)

    def Lie_update_SE3_B(self, t, q, Delta_s):
        q_nodes = q.reshape(self.nnodes, self.nq_node)
        Delta_s_nodes = Delta_s.reshape(self.nnodes, 6)

        # bring position update to B-system
        A_IB0 = self._Exp_SO3(q_nodes[:, 3:])
        H_IB0 = SE3(A_IB0, q_nodes[:, :3])
        Delta_s_nodes[:, :3] = np.einsum("ijk,ij->ik", A_IB0, Delta_s_nodes[:, :3])

        # SE3 update
        H_B0B1 = Exp_SE3(Delta_s_nodes)
        H_IB1 = H_IB0 @ H_B0B1

        # extract new q
        q_nodes[:, :3] = H_IB1[:, :3, 3]
        q_nodes[:, 3:] = self._Log_SO3(H_IB1[:, :3, :3])

        return q_nodes.reshape(-1)

    def Lie_update_SE3_I(self, t, q, Delta_s):
        q_nodes = q.reshape(self.nnodes, self.nq_node)
        Delta_s_nodes = Delta_s.reshape(self.nnodes, 6)

        # bring rotation update to I-system
        A_IB = self._Exp_SO3(q_nodes[:, 3:])
        H_IB = SE3(A_IB, q_nodes[:, :3])
        Delta_s_nodes[:, 3:] = np.einsum("ijk,ik->ij", A_IB, Delta_s_nodes[:, 3:])

        # SE3 update
        H_I2I = Exp_SE3(Delta_s_nodes)
        H_I2B = H_I2I @ H_IB

        # extract new q
        q_nodes[:, :3] = H_I2B[:, :3, 3]
        q_nodes[:, 3:] = self._Log_SO3(H_I2B[:, :3, :3])

        return q_nodes.reshape(-1)
