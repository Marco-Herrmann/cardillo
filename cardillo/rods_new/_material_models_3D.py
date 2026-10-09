from collections import namedtuple

from cardillo.rods.discretization.gauss import gauss
from cardillo.math.algebra import ax2skew

from .CableSlabMaterials.Elasticity import Elasticity as Elasticity3D
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


class Elasticity(RodMaterialModel):
    def __init__(self, E, nu, quadrature):
        self.material_3D = Elasticity3D(E, nu)
        self.quadrature = quadrature

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
                B_r_CPj = self.quadrature.B_r_CP[j]

                # TODO: this is constant per quadrature point!
                r_tilde = ax2skew(B_r_CPj)
                P_bar = np.zeros((6, 6))
                P_bar[[0, 3, 4], :3] = np.eye(3)
                P_bar[[0, 3, 4], 3:] = -r_tilde

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

    def sigma(self, epsilon, epsilon0, prepare):
        return self.get_stress(epsilon, epsilon0, prepare)[0]

    def sigma_epsilon(self, epsilon, epsilon0, prepare):
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
                lambda epsilon_: self.sigma(epsilon_, np.atleast_2d(epsilon0[i]), prepare),
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
