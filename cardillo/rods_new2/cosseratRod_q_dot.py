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


class CosseratRod_kin_constraints(ABC):
    def __init__(self, parent, parametrization, projection):
        self.parent = parent
        self.nnodes = self.parent.nnodes

        # TODO: move T_IB_inv and T_IB_inv_P to rP_dot_from_vO_IB class!, as it is only relevant there!
        assert parametrization in ["Quaternion", "R12"]
        if parametrization == "Quaternion":
            self.nq_node = 7

            self._T_IB_inv = T_SO3_inv_quat
            self._T_IB_inv_P = T_SO3_inv_quat_P(None)  # evaluate as it is constant

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

            self._T_IB_inv = T_SO3_inv_R9
            self._T_IB_inv_P = T_SO3_inv_R9_R9(None)  # evaluate as it is constant

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
        coo = CooMatrix((self.parent.nla_S, self.parent.nq))
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

        coo = CooMatrix((self.parent.nla_S, self.parent.nq))
        coo.data = data.ravel()

        coo.row = self._g_S_q_row
        coo.col = self._g_S_q_col
        return coo

        from cardillo.math.approx_fprime import approx_fprime

        g_q_num = approx_fprime(q.copy(), lambda q_: self.g_R9(t, q_))
        diff = coo.toarray() - g_q_num
        print(np.max(np.abs(diff)))
        return g_q_num


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


class CosseratRod_rP_dot_from_vO_IB(CosseratRod_kin_constraints):
    def __init__(self, parent, parametrization, projection):
        super().__init__(parent, parametrization, projection)
        self.parent.q_dot = self.q_dot
        self.parent.q_dot_u = self.q_dot_u
        self.parent.q_dot_q = self.q_dot_q

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
        # qnodes = q.reshape(self.nnodes, self.nq_node)
        unodes = u.reshape(self.nnodes, 6)

        blocks = np.empty((self.nnodes, self.nq_node, self.nq_node))
        blocks[:, :3] = 0.0
        blocks[:, 3:, :3] = 0.0
        blocks[:, 3:, 3:] = np.einsum(
            "jkl,ik->ijl",
            self._T_IB_inv_P,
            unodes[:, 3:],
        )

        return csr_array(block_diag(blocks))

    def q_dot_u(self, t, q):
        qnodes = q.reshape(self.nnodes, self.nq_node)

        blocks = np.empty((self.nnodes, self.nq_node, 6))
        blocks[:, :3, :3] = eye3
        blocks[:, :3, 3:] = 0.0
        blocks[:, 3:, :3] = 0.0
        blocks[:, 3:, 3:] = self._T_IB_inv(qnodes[:, 3:])

        return csr_array(block_diag(blocks))  # this keeps the 0's
