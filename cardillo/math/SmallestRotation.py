import numpy as np
from warnings import warn

from cardillo.math import (
    e1,
    e2,
    e3,
    smallest_rotation,
    norm,
    skew2ax,
)
from cardillo.math.approx_fprime import approx_fprime


class SmallestRotation:
    def __init__(self, R_ei_J: np.ndarray, i: int = 1):
        """Smallest rotation matrix from R_ei_J to R_ei_R

        Parameters
        ----------
        R_ei_J : np.ndarray (3,)
            direction of ei_J, represented in the reference basis R,
            must be a unitvector, otherwise the calculation is wrong
        i : int
            index of the reference basis vector in {1, 2, 3}

        Properties
        ----------
        A_RJ : np.ndarray (3, 3)
            transformation matrix from J to R such that R_ei_J = A_RJ @ J_ei_J
        A_RJ_e : np.ndarray (3, 3, 3)
            derivative of A_RJ with respect to R_ei_J
        """

        # save arguments for derivative
        self.i_py = i - 1
        self.R_ei_J = R_ei_J

        # indices for the other axis
        self.idx_1 = (self.i_py + 1) % 3
        self.idx_2 = (self.i_py + 2) % 3

        # get some useful values
        self.cos_phi = self.R_ei_J[self.i_py]
        self.denom_inv = 1.0 / (1.0 + self.cos_phi)

        #######################
        # cos_phi * np.eye(3) #
        #######################
        self.A_RJ = self.cos_phi * np.eye(3, dtype=float)
        #####################
        # ax2skew(self.phi) #
        #####################
        self.A_RJ[self.i_py, self.idx_1] = -R_ei_J[self.idx_1]
        self.A_RJ[self.i_py, self.idx_2] = -R_ei_J[self.idx_2]
        self.A_RJ[self.idx_1, self.i_py] = R_ei_J[self.idx_1]
        self.A_RJ[self.idx_2, self.i_py] = R_ei_J[self.idx_2]
        ##############################
        # np.outer(phi, phi) / denom #
        ##############################
        # fmt: off
        self.A_RJ[self.idx_1, self.idx_1] += R_ei_J[self.idx_2] * R_ei_J[self.idx_2] * self.denom_inv
        self.A_RJ[self.idx_2, self.idx_2] += R_ei_J[self.idx_1] * R_ei_J[self.idx_1] * self.denom_inv
        self.A_RJ[self.idx_1, self.idx_2] = self.A_RJ[self.idx_2, self.idx_1] = (
            -R_ei_J[self.idx_1] * R_ei_J[self.idx_2] * self.denom_inv
        )
        # fmt: on

        if self.cos_phi < 1e-6 - 1:
            warn("SmallestRotation: Denominator is close to zero!")
            # TODO: handle this case

    @property
    def A_RJ_e(self):
        # get some useful values
        denom_inv2 = self.denom_inv * self.denom_inv

        A_RJ_e = np.zeros((3, 3, 3), dtype=float)
        #######################
        # cos_phi * np.eye(3) #
        #######################
        A_RJ_e[:, :, self.i_py] = np.eye(3, dtype=float)
        #####################
        # ax2skew(self.phi) #
        #####################
        A_RJ_e[self.i_py, self.idx_1, self.idx_1] = -1.0
        A_RJ_e[self.i_py, self.idx_2, self.idx_2] = -1.0
        A_RJ_e[self.idx_1, self.i_py, self.idx_1] = 1.0
        A_RJ_e[self.idx_2, self.i_py, self.idx_2] = 1.0
        ##############################
        # np.outer(phi, phi) / denom #
        ##############################
        # fmt: off
        A_RJ_e[self.idx_1, self.idx_1, self.idx_2] = 2 * self.R_ei_J[self.idx_2] * self.denom_inv
        A_RJ_e[self.idx_2, self.idx_2, self.idx_1] = 2 * self.R_ei_J[self.idx_1] * self.denom_inv
        A_RJ_e[self.idx_1, self.idx_2, self.idx_1] = A_RJ_e[self.idx_2, self.idx_1, self.idx_1] =(
            - self.R_ei_J[self.idx_2] * self.denom_inv
        )
        A_RJ_e[self.idx_1, self.idx_2, self.idx_2] = A_RJ_e[self.idx_2, self.idx_1, self.idx_2] =(
            - self.R_ei_J[self.idx_1] * self.denom_inv
        )
        A_RJ_e[self.idx_1, self.idx_1, self.i_py] -= self.R_ei_J[self.idx_2] * self.R_ei_J[self.idx_2] * denom_inv2
        A_RJ_e[self.idx_2, self.idx_2, self.i_py] -= self.R_ei_J[self.idx_1] * self.R_ei_J[self.idx_1] * denom_inv2
        A_RJ_e[self.idx_1, self.idx_2, self.i_py] = A_RJ_e[self.idx_2, self.idx_1, self.i_py] = (
            self.R_ei_J[self.idx_1] * self.R_ei_J[self.idx_2] * denom_inv2
        )
        # fmt: on

        return A_RJ_e

    @property
    def A_RJ_e_num(self):
        return approx_fprime(
            self.R_ei_J,
            lambda R_ei_J: SmallestRotation(R_ei_J, self.i_py + 1).A_RJ,
            method="2-point",
            eps=1e-8,
        )

    @property
    def T_RJ(self):
        T = np.zeros((3, 3))
        T[self.i_py, self.idx_1] = self.denom_inv * self.R_ei_J[self.idx_2]
        T[self.i_py, self.idx_2] = -self.denom_inv * self.R_ei_J[self.idx_1]

        T[self.idx_1, self.i_py] = self.denom_inv * self.R_ei_J[self.idx_2]
        T[self.idx_1, self.idx_2] = -1.0

        T[self.idx_2, self.i_py] = -self.denom_inv * self.R_ei_J[self.idx_1]
        T[self.idx_2, self.idx_1] = 1.0

        return T

    def R_kappa_RJ_num(self, dR_ei_J: np.ndarray):
        # dR_ei_J = dR_ei_J - R_u * (R_u @ dR_ei_J)

        A_RJ_xi = np.einsum("ijk,k->ij", SR.A_RJ_e, dR_ei_J)
        curv_tilde = A_RJ.T @ A_RJ_xi

        error = np.linalg.norm(curv_tilde + curv_tilde.T)
        if error > 1e-6:
            print(f"symmetric part: {error}")

        return skew2ax(curv_tilde)

    def R_kappa_RJ(self, dR_ei_J: np.ndarray):
        kappa_i = self.denom_inv * (
            self.R_ei_J[self.idx_2] * dR_ei_J[self.idx_1]
            - self.R_ei_J[self.idx_1] * dR_ei_J[self.idx_2]
        )
        kappa_12 = dR_ei_J - dR_ei_J[self.i_py] * self.denom_inv * self.R_ei_J

        kappa = np.empty(3)
        kappa[self.i_py] = kappa_i
        kappa[self.idx_1] = -kappa_12[self.idx_2]
        kappa[self.idx_2] = kappa_12[self.idx_1]

        curv = self.R_kappa_RJ_num(dR_ei_J)
        error = np.linalg.norm(kappa - curv)
        if error > 1e-10:
            print(f"error: {error}")

        return kappa


if __name__ == "__main__":
    for _ in range(100):
        R_u = np.random.rand(3)
        i = np.random.randint(1, 4)

        if i == 1:
            R_ei_R = e1
        elif i == 2:
            R_ei_R = e2
        elif i == 3:
            R_ei_R = e3

        R_u = R_u / norm(R_u)

        A_RJ = smallest_rotation(R_ei_R, R_u, normalize=False)
        SR = SmallestRotation(R_u, i=i)

        diff_trafo = np.linalg.norm(A_RJ - SR.A_RJ)
        diff_deriv = np.linalg.norm(SR.A_RJ_e_num - SR.A_RJ_e)

        dR_ei_J = np.random.rand(3)
        dR_ei_J = dR_ei_J - R_u * (R_u @ dR_ei_J)

        kappa_a = SR.R_kappa_RJ(dR_ei_J)
        curv = SR.T_RJ @ dR_ei_J
        diff_kappa = np.linalg.norm(kappa_a - curv)

        print(
            f"diff trafo : {diff_trafo:.2e}, diff derivative: {diff_deriv:.2e}, diff kappa: {diff_kappa:.2e}"
        )
