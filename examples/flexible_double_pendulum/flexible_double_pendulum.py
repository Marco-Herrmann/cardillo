from cProfile import Profile
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.constraints import RigidConnection, Revolute
from cardillo.forces import B_Moment, Force
from cardillo.math import e2, e3
from cardillo.math.rotations import A_IB_basic
from cardillo.rods import (
    CircularCrossSection,
    RectangularCrossSection,
    CrossSectionInertias,
    CrossSectionInertias_new,
    animate_beam,
)
from cardillo.rods._material_models_new import Simo1986

from cardillo.solver import Newton, SolverOptions, ScipyDAE, DualStormerVerlet
from cardillo.rods.boostedCosseratRod import make_BoostedCosseratRod

from cardillo.discrete import RigidBody, Sphere, Box, Frame
from cardillo.forces import Force
from cardillo.contacts import Sphere2Plane, Sphere2Sphere


def flexible_double_pendulum(Rod, show_plots, name):
    length = 1.0
    width_y = 0.05
    width_z = 0.05
    nelements = 10

    rho = 7000.0
    cross_section = RectangularCrossSection(width_y, width_z)
    A = cross_section.area
    B_I = cross_section.second_moment
    cross_section_inertias = CrossSectionInertias_new(
        A_rho0=A * rho, B_I_rho0=B_I * rho
    )

    # material properties
    E = 210_000_000
    mu = 0.3
    G = E / (2 * (1 + mu))
    A = cross_section.area
    I = np.diag(cross_section.second_moment)
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * I[0], E * I[1], E * I[2]])

    material_model = Simo1986(Ei, Fi)

    # initialize system
    system = System()

    #####
    # rod
    #####
    A_IB0 = A_IB_basic(np.pi / 2 * 0.0).y
    r_OP0 = np.zeros(3, dtype=float)
    r_OP1 = r_OP0 + A_IB0 @ np.array([length, 0.0, 0.0])

    g = np.array([0.0, 0.0, -9.81 * A * rho])
    b = lambda t, xis: g if isinstance(xis, (float, int)) else np.array([g] * len(xis))

    # compute straight initial configuration of cantilever
    q0s = [
        Rod.straight_configuration(nelements, length, r_OP0, A_IB0),
        Rod.straight_configuration(nelements, length, r_OP1, A_IB0),
    ]

    # construct cantilever
    rods = [
        Rod(
            cross_section,
            material_model,
            nelements,
            Q=q0i,
            q0=q0i,
            distributed_load=[b, None],
            cross_section_inertias=cross_section_inertias,
            name=f"Rod{i}",
        )
        for i, q0i in enumerate(q0s)
    ]

    # initialize system
    system = System()
    system.add(*rods)

    joint0 = Revolute(rods[0], system.origin, 1, xi1=0.0, name="rev_origin_rod0")
    joint1 = Revolute(rods[1], rods[0], 1, xi1=0.0, xi2=1.0, name="rev_rod0_rod1")
    system.add(joint0, joint1)

    ###########
    # contact #
    ###########
    w = 4.5
    floor = Box(Frame)(
        dimensions=[w, w, 0.0001],
        name="floor",
        A_IB=A_IB_basic(np.pi / 2).y,
    )
    system.add(floor)  # (only for visualization purposes)

    for i in range(1, 2):
        for xi in np.linspace(0, 1, nelements + 1)[-1:]:
            system.add(
                Sphere2Plane(
                    floor,
                    rods[i],
                    xi=xi,
                    mu=mu,
                    r=width_z,
                    e_N=0.0,
                    e_F=0.0,
                    name="floor2",
                )
            )

    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    t1 = 1.5
    dt = 5e-3
    dt = 1e-2

    # solver = ScipyDAE(system, t1, dt)
    solver = DualStormerVerlet(
        system, t1, dt, options=SolverOptions(newton_max_iter=100)
    )
    prof = Profile()
    prof.enable()
    sol = solver.solve()
    prof.disable()

    E_pot = np.zeros(len(sol.t))
    E_kin = np.zeros(len(sol.t))
    for i in range(len(sol.t)):
        E_pot[i] = system.E_pot(sol.t[i], sol.q[i])
        E_kin[i] = system.E_kin(sol.t[i], sol.q[i], sol.u[i])

    fig, ax = plt.subplots()
    ax.plot(sol.t, E_pot)
    ax.plot(sol.t, E_kin)
    ax.plot(sol.t, E_kin + E_pot)

    # save solution
    path = Path(__file__)
    sol.save(Path(path.parent, f"sol_{name}.pkl"))

    # to view: run "view snakeviz.exe .\rod.prof" in shell
    prof.dump_stats(Path(path.parent, f"prof_{name}.prof"))

    fig, ax, anim = animate_beam(
        sol.t, sol.q, rods, scale=2 * length, scale_di=width_y, show=False
    )

    # draw surface for collision
    Y_x = np.linspace(-w / 2, w / 2, num=2)
    Z_x = np.linspace(-w / 2, w / 2, num=2)
    Y_x, Z_x = np.meshgrid(Y_x, Z_x)
    X_x = np.zeros_like(Y_x)
    ax.plot_surface(X_x, Y_x, Z_x, alpha=0.2)

    if show_plots:
        plt.show()


if __name__ == "__main__":
    pDeg = 2
    flexible_double_pendulum(
        make_BoostedCosseratRod(
            polynomial_degree=pDeg,
            quadrature_dyn=(pDeg + 1, "Trapezoidal"),
            quadrature_ext=(pDeg + 1, "Trapezoidal"),
        ),
        show_plots=True,
        name="boosted",
    )
