import matplotlib.pyplot as plt
import numpy as np
from timeit import timeit

from cardillo import System
from cardillo.constraints import RigidConnection
from cardillo.rods import (
    CircularCrossSection,
    CrossSectionInertias_new,
    animate_beam,
)
from cardillo.rods._material_models_new import Simo1986
from cardillo.rods.KirchhoffLoveRod import make_KirchhoffLoveRod
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.rods.boostedCosseratRod import make_BoostedCosseratRod
from cardillo.math.rotations import Exp_SO3_quat
from cardillo.math.approx_fprime import approx_fprime
from cardillo.solver import SolverOptions, Newton
from cardillo.forces import Force, B_Moment

if __name__ == "__main__":
    constitutive_law = Simo1986(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    constitutive_law = Simo1986(
        np.array([1.0 * 0.1, 10.0, 10.0]), np.array([4.0, 5.0, 5.0])
    )
    cross_section = CircularCrossSection(0.1)
    A_rho0 = np.random.rand()
    B_I_rho0 = np.diag(np.random.rand(3))
    cross_section_inertia = CrossSectionInertias_new(A_rho0=A_rho0, B_I_rho0=B_I_rho0)
    cross_section_inertia = False
    nelement = 8

    Rod = make_KirchhoffLoveRod()
    # Rod = make_BoostedCosseratRod()

    Q = Rod.straight_configuration(nelement, 5)
    q0 = Q.copy()
    # q0[14] += 0.1
    rod = Rod(
        cross_section,
        constitutive_law,
        nelement,
        Q=Q,
        q0=q0,
        cross_section_inertias=cross_section_inertia,
    )

    # np.set_printoptions(linewidth=300, precision=3)
    # print(rod.h(0.0, q0, rod.u0))
    # print(rod.h_q(0.0, q0, rod.u0))

    system = System()
    constraint = RigidConnection(system.origin, rod, xi2=0.0)
    forcing = Force(lambda t: np.array([0.5, 1.0, 0.2]) * t * 0.5, rod, xi=1.0)
    forcing = Force(lambda t: np.array([0.0, 1.0, 0.0]) * t, rod, xi=1.0)
    forcing = Force(lambda t: np.array([0.0, 0.0, 1.0]) * t, rod, xi=1.0)
    forcing = Force(lambda t: np.array([0.0, 0.0, 1.0]) * 0.0, rod, xi=1.0)
    # TODO: why moment in y and z not working?
    M = 2 * np.pi
    print(M)
    forcing = B_Moment(lambda t: np.array([0.0, 0.0, M]) * t, rod, xi=1.0)

    system.add(rod, constraint, forcing)

    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    # print(constraint.g(0.0, q0[constraint.qDOF]))
    # print(constraint.g_q(0.0, q0[constraint.qDOF]))
    # print(constraint.W_g(0.0, q0[constraint.qDOF]))
    # print(constraint.Wla_g_q(0.0, q0[constraint.qDOF], np.random.rand(6)))

    solver = Newton(system, n_load_steps=100)
    sol = solver.solve()

    # Li = 5.0
    # for i in range(len(sol.q)):
    #     sol.q[i] = np.array([
    #         0.0, 0.0, 0.0,
    #         Li / np.sqrt(2), Li / np.sqrt(2), 0.0,
    #         np.sqrt(Li), 0.0, 0.0, 0.0,
    #         np.sqrt(Li) / np.sqrt(2), 0.0, 0.0, np.sqrt(Li) / np.sqrt(2),
    #         0.0,
    #     ])

    # rod.h(0.0, sol.q[-1], None)

    # xis, eps_Ga, eps_Ka = rod.eval_strains(0.0, sol.q[-1], None, None, n_per_element=50)
    xis, eps_Ga, eps_Ka = rod.eval_strains(
        0.0, sol.q[-1], sol.la_c[-1], sol.la_g[-1], n_per_element=50
    )

    fig, ax = plt.subplots(3, 2)
    for i in range(3):
        ax[i, 0].plot(xis, eps_Ga[:, i])
        ax[i, 1].plot(xis, eps_Ka[:, i])
    # plt.show()

    fig, ax = plt.subplots(2, 2)
    r_OP0 = sol.q[:, 0:3]
    r_OP1 = sol.q[:, 3:6]
    P0 = sol.q[:, 6:10]
    P1 = sol.q[:, 10:14]
    alpha = sol.q[:, 14]

    ax[0, 0].plot(sol.t, r_OP0)
    ax[1, 0].plot(sol.t, r_OP1)

    ax[0, 1].plot(sol.t, P0 * P0)  # - r_OP1[:, 0][:, None])
    ax[1, 1].plot(sol.t, P1 * P1)  # - r_OP1[:, 0][:, None])

    # animation
    # animate_beam(np.linspace(0, 1, 5), np.array([system.q0] * 5), [rod], scale=5, n_r=50, n_frames=11)
    animate_beam(sol.t, sol.q, [rod], scale=5, n_r=50, n_frames=11)

    print("solved")
