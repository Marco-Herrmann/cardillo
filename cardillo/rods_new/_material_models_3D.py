from collections import namedtuple
import numpy as np

from cardillo.rods.discretization.gauss import gauss
from cardillo.math.algebra import ax2skew

from ._material_models import RodMaterialModel

pyfem_point = namedtuple("pyfem_point", ["strain", "dstrain"])


class Rectangle_Quadrature:
    def __init__(self, ny, nz, ay, az):
        self.n = ny * nz
        ry, wy = gauss(ny, [-ay / 2, ay / 2])
        rz, wz = gauss(nz, [-az / 2, az / 2])

        rX_ = np.zeros(self.n)
        rY, rZ = np.meshgrid(ry, rz)
        self.B_r_CP = np.column_stack([rX_, rY.ravel(), rZ.ravel()])

        wY, wZ = np.meshgrid(wy, wz)
        self.weights = (wY * wZ).ravel()


class Material3D(RodMaterialModel):
    def __init__(self, material_3D, quadrature, constant_C=False, eps_fd=1e-8):
        """
        Rod material model from a 3D material law, integrated over the cross
        section with the given quadrature.

        Parameters
        ----------
        material_3D : 3D material model with getStress(kinematics, actions, element, intpointID)
        quadrature : cross-section quadrature, e.g., Rectangle_Quadrature
        constant_C : bool
            The tangent of the 3D material law is constant, e.g., for linear
            elasticity. Then, C(epsilon) is only evaluated once by the system.
        eps_fd : float
            Relative step size of the forward differences in Csigma_epsilon,
            as the 3D material laws only provide the tangent.
        """
        self.material_3D = material_3D
        self.quadrature = quadrature
        self.constant_C = constant_C
        self.eps_fd = eps_fd

        # project (B_gamma, B_kappa) to 3D strain, constant per quadrature point
        self.P_bar = np.zeros((quadrature.n, 6, 6))
        self.P_bar[:, [0, 5, 4], :3] = np.eye(3)
        self.P_bar[:, [0, 5, 4], 3:] = -ax2skew(quadrature.B_r_CP)

    def prepare(self, xi):
        return dict(n=len(xi))

    def potential(self, epsilon, epsilon0, prepare): ...

    # TODO: cache! based on epsilon
    def get_stress(self, epsilon, epsilon0, prepare):
        d_epsilon = epsilon - epsilon0
        n = prepare["n"]
        sigma = np.zeros((n, 6))
        sigma_epsilon = np.zeros((n, 6, 6))
        for i in range(n):
            for j in range(self.quadrature.n):
                P_bar = self.P_bar[j]

                # project (B_gamma, B_kappa) to 3D strain
                eps6 = P_bar @ d_epsilon[i]
                kinematics = pyfem_point(eps6, None)

                # compute 3D stress and tangent stiffness
                sig6, sig6_eps6 = self.material_3D.getStress(kinematics, None, i, j)

                # project 3D stress to (B_n, B_m)
                wj = self.quadrature.weights[j]
                sigma[i] += P_bar.T @ sig6 * wj
                sigma_epsilon[i] += P_bar.T @ sig6_eps6 @ P_bar * wj

        return sigma, sigma_epsilon

    # compliance form: C(epsilon) @ sigma - epsilon_rel(epsilon) = 0,
    # with C = sigma_epsilon^-1 and epsilon_rel = C @ sigma
    def epsilon_rel(self, epsilon, epsilon0, prepare):
        sigma, sigma_epsilon = self.get_stress(epsilon, epsilon0, prepare)
        return np.linalg.solve(sigma_epsilon, sigma[:, :, None])[:, :, 0]

    def epsilon_rel_epsilon(self, epsilon, epsilon0, prepare):
        # d(C @ sigma)/d epsilon = C @ sigma_epsilon + d(C @ sigma)/d epsilon|_sigma
        sigma = self.get_stress(epsilon, epsilon0, prepare)[0]
        epsilon_rel_epsilon = self.Csigma_epsilon(epsilon, epsilon0, sigma, prepare)
        epsilon_rel_epsilon[:, range(6), range(6)] += 1.0
        return epsilon_rel_epsilon

    def C(self, epsilon, epsilon0, prepare):
        sigma_epsilon = self.get_stress(epsilon, epsilon0, prepare)[1]
        return np.linalg.inv(sigma_epsilon)

    def Csigma_epsilon(self, epsilon, epsilon0, sigma, prepare):
        # d(K^-1 @ sigma)/d epsilon_k = -K^-1 @ dK/d epsilon_k @ v, with
        # K = sigma_epsilon, v = K^-1 @ sigma and dK/d epsilon_k by forward
        # differences, as the 3D material laws only provide K
        Csigma_epsilon = np.zeros((len(epsilon), 6, 6))
        if self.constant_C:
            return Csigma_epsilon

        K = self.get_stress(epsilon, epsilon0, prepare)[1]
        C = np.linalg.inv(K)
        v = np.einsum("ijk,ik->ij", C, sigma)
        for k in range(6):
            h = self.eps_fd * (1 + np.abs(epsilon[:, k]))
            epsilon_h = epsilon.copy()
            epsilon_h[:, k] += h
            K_h = self.get_stress(epsilon_h, epsilon0, prepare)[1]
            K_k_v = np.einsum("ijk,ik->ij", K_h - K, v) / h[:, None]
            Csigma_epsilon[:, :, k] = -np.einsum("ijk,ik->ij", C, K_k_v)
        return Csigma_epsilon

    def epsilon_rel_inv(self, sigma, epsilon0, prepare, rtol=1e-12, max_iter=20):
        # solves C(epsilon) @ sigma - epsilon_rel(epsilon) = 0 for epsilon,
        # which is equivalent to sigma(epsilon) - sigma = 0 as C is regular
        epsilon = epsilon0.copy()
        for _ in range(max_iter):
            sigma_eps, K = self.get_stress(epsilon, epsilon0, prepare)
            R = sigma_eps - sigma
            if np.max(np.abs(R) / (1 + np.abs(sigma))) < rtol:
                break
            epsilon -= np.linalg.solve(K, R[:, :, None])[:, :, 0]
        else:
            raise RuntimeError("epsilon_rel_inv: Newton iteration did not converge")

        return epsilon

    # displacement based
    def sigma(self, epsilon, epsilon0, epsilon_dot, prepare):
        return self.get_stress(epsilon, epsilon0, prepare)[0]

    def sigma_epsilon(self, epsilon, epsilon0, epsilon_dot, prepare):
        sigma_epsilon = self.get_stress(epsilon, epsilon0, prepare)[1]
        return (
            sigma_epsilon[:, :3, :3],  # d(B_n) / d(B_gamma)
            sigma_epsilon[:, :3, 3:],  # d(B_n) / d(B_kappa)
            sigma_epsilon[:, 3:, :3],  # d(B_m) / d(B_gamma)
            sigma_epsilon[:, 3:, 3:],  # d(B_m) / d(B_kappa)
        )

        from cardillo.math.approx_fprime import approx_fprime

        # numerical derivative
        sigma_epsilon_num = np.zeros((*epsilon.shape, 6))
        for i, epsiloni in enumerate(epsilon):
            sigma_epsilon_num[i] = approx_fprime(
                np.atleast_2d(epsiloni),
                lambda epsilon_: self.sigma(
                    epsilon_, np.atleast_2d(epsilon0[i]), prepare
                ),
            )
        error = np.linalg.norm(sigma_epsilon_num - sigma_epsilon)
        max_val = np.max(np.abs(sigma_epsilon_num))
        print(f"\nDiff: {error:.3e}, Diff_rel: {error / max_val:.3e}")
        return (
            sigma_epsilon[:, :3, :3],
            sigma_epsilon[:, :3, 3:],
            sigma_epsilon[:, 3:, :3],
            sigma_epsilon[:, 3:, 3:],
        )
