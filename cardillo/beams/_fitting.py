import numpy as np

from scipy.optimize import minimize
from scipy.optimize import least_squares
from cardillo.math import SE3, SE3inv, Log_SE3, Log_SE3_H


# TODO: Add clamping for first and last node
def fit_configuration(
    rod,
    r_OPs,
    A_IKs,
    use_cord_length=False,
    nodal_cDOF=[0, -1],
    verbose=0,
):
    # number of sample points
    n_samples = len(r_OPs)

    # compute homogeneous transformations for target points
    Hs = np.array([SE3(A_IKs[i], r_OPs[i]) for i in range(n_samples)])

    # cord-length of relative twists
    if use_cord_length:
        relative_twists = np.array(
            [Log_SE3(SE3inv(Hs[i - 1]) @ Hs[i]) for i in range(1, n_samples)]
        )
        segment_lengths = np.linalg.norm(relative_twists, axis=1)
        cord_lengths = np.cumsum(segment_lengths)
        xis = np.zeros(n_samples, dtype=float)
        xis[1:] = cord_lengths / cord_lengths[-1]
    else:
        # linear spaced evaluation points for the rod
        xis = np.linspace(0, 1, n_samples)

    # initial vector of generalized coordinates
    Z0 = np.zeros(rod.nq, dtype=float)

    ###########
    # warmstart
    ###########
    # initialized nodal unknowns with closest data w.r.t. xi values
    xi_idx_r = np.array(
        [
            np.where(xis >= rod.mesh_r.knot_vector.data[i])[0][0]
            for i in range(rod.mesh_r.nnodes)
        ]
    )
    xi_idx_psi = np.array(
        [
            np.where(xis >= rod.mesh_psi.knot_vector.data[i])[0][0]
            for i in range(rod.mesh_psi.nnodes)
        ]
    )

    for node, xi_idx in enumerate(xi_idx_r):
        Z0[rod.nodalDOF_r[node]] = r_OPs[xi_idx]
    for node, xi_idx in enumerate(xi_idx_psi):
        Z0[rod.nodalDOF_psi[node]] = rod.RotationBase.Log_SO3(A_IKs[xi_idx])

    # constrained nodes
    zDOF = np.arange(rod.nq)
    cDOF_r = rod.nodalDOF_r[nodal_cDOF]
    cDOF_psi = rod.nodalDOF_psi[nodal_cDOF]
    cDOF = np.concatenate((cDOF_r.flatten(), cDOF_psi.flatten()))
    fDOF = np.setdiff1d(zDOF, cDOF)

    # generate redundant coordinates
    Z0_boundary_r = Z0[cDOF_r]
    Z0_boundary_psi = Z0[cDOF_psi]

    def make_redundant_coordinates(q):
        z = np.zeros(rod.nq, dtype=q.dtype)
        z[fDOF] = q
        z[cDOF_r] = Z0_boundary_r
        z[cDOF_psi] = Z0_boundary_psi
        return z

    def residual(q):
        z = make_redundant_coordinates(q)

        R = np.zeros(6 * n_samples, dtype=z.dtype)
        for i, xi in enumerate(xis):
            # find element number and extract elemt degrees of freedom
            el = rod.element_number(xi)
            elDOF = rod.elDOF[el]
            ze = z[elDOF]

            # interpolate position and orientation
            r_OP, A_IK, _, _ = rod._eval(ze, xi)

            # compute homogeneous transformation
            H = SE3(A_IK, r_OP)

            # use relative twist as error measure
            R[6 * i : 6 * (i + 1)] = Log_SE3(SE3inv(Hs[i]) @ H)

        return R

    def jac(q):
        z = make_redundant_coordinates(q)

        J = np.zeros((6 * n_samples, len(z)), dtype=z.dtype)
        for i, xi in enumerate(xis):
            # find element number and extract elemt degrees of freedom
            el = rod.element_number(xi)
            elDOF = rod.elDOF[el]
            ze = z[elDOF]
            H_qe = np.zeros((4, 4, len(ze)))

            # interpolate position and orientation
            (
                r_OP,
                A_IK,
                _,
                _,
                r_OP_ze,
                A_IK_ze,
                _,
                _,
            ) = rod._deval(ze, xi)

            # compute homogeneous transformation
            H = SE3(A_IK, r_OP)

            Log_invHsH_H = Log_SE3_H(SE3inv(Hs[i]) @ H)

            H_qe[:3, :3, :] = A_IK_ze
            H_qe[:3, 3, :] = r_OP_ze

            # compute relative rotation vector
            # psi_rel_A_rel = Log_SO3_A(r_OPs[i].T @ A_IK)
            # psi_rel_qe = np.einsum("ikl,mk,mlj->ij", psi_rel_A_rel, r_OPs[i], A_IK_qe)

            invHsH_H = np.einsum("ij,jl,km->iklm", SE3inv(Hs[i]), np.eye(4), np.eye(4))

            # insert to residual
            J[6 * i : 6 * (i + 1), elDOF] = np.einsum(
                "ikl,klmn,mno->io", Log_invHsH_H, invHsH_H, H_qe
            )

            # def psi_rel(qe):
            #     # evaluate shape functions
            #     N = mesh.eval_basis(xii)

            #     # interpoalte rotations
            #     A_IK = np.zeros((3, 3), dtype=q.dtype)
            #     for node in range(mesh.nnodes_per_element):
            #         A_IK += N[node] * Exp_SO3(qe[mesh.nodalDOF_element[node]])

            #     # compute relative rotation vector
            #     return Log_SO3(A_IKs[i].T @ A_IK)

            # psi_rel_qe_num = approx_fprime(qe, psi_rel)
            # diff = psi_rel_qe - psi_rel_qe_num
            # error = np.linalg.norm(diff)
            # print(f"error psi_rel_qe: {error}")

            # # insert to residual
            # J[3 * i:3 * (i + 1), elDOF] = psi_rel_qe_num

        # jacobian w.r.t. to minimal coordinates
        J = J[:, fDOF]
        return J

        # from cardillo.math import approx_fprime
        # J_num = approx_fprime(q, residual, method="cs")
        # diff = J - J_num
        # error = np.linalg.norm(diff)
        # print(f"error J: {error}")
        # return J_num
        # # return J

    res = least_squares(
        residual,
        Z0[fDOF],
        jac=jac,
        # jac="2-point",
        method="trf",
        # method="lm",
        # verbose=verbose,
        verbose=0,
    )
    Z0 = res.x
    Z0 = make_redundant_coordinates(Z0)
    print(f"res: {res}")

    rod.set_initial_strains(Z0)
    return Z0
