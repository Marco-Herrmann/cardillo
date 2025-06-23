from math import pi
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
    Simo1986,
    animate_beam,
)
from cardillo.rods.force_line_distributed import Force_line_distributed
from cardillo.rods.cosseratRod import (
    make_CosseratRod,
)
from cardillo.solver import Newton, SolverOptions, ScipyDAE, DualStormerVerlet
from cardillo.rods.boostedCosseratRod import make_BoostedCosseratRod

from cardillo.discrete import RigidBody, Sphere, Box, Frame
from cardillo.forces import Force
from cardillo.contacts import Sphere2Plane, Sphere2Sphere


def flexible_double_pendulum(Rod, show_plots, name):
    length = 1.0
    width_y = 0.05
    width_z = 0.05
    nelements = 4

    rho = 7000.0
    cross_section = RectangularCrossSection(width_y, width_z)
    cross_section_inertias = CrossSectionInertias(rho, cross_section)

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
            cross_section_inertias=cross_section_inertias,
            name=f"Rod{i}",
        )
        for i, q0i in enumerate(q0s)
    ]

    #
    gravities = [Force_line_distributed(g, rod) for rod in rods]

    # initialize system
    system = System()
    system.add(*rods, *gravities)

    joint0 = Revolute(rods[0], system.origin, 1, xi1=0.0)
    joint1 = Revolute(rods[1], rods[0], 1, xi1=0.0, xi2=1.0)
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
        for xi in np.linspace(0, 1, nelements + 1)[1:]:
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

    system.assemble()

    t1 = 1.5
    dt = 5e-3
    dt = 1e-2

    # solver = ScipyDAE(system, t1, dt)
    solver = DualStormerVerlet(
        system, t1, dt, options=SolverOptions(newton_max_iter=100)
    )
    sol = solver.solve()

    # save solution
    path = Path(__file__)
    sol.save(Path(path.parent, f"sol_{name}.pkl"))

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
    flexible_double_pendulum(
        make_BoostedCosseratRod(polynomial_degree=2), show_plots=False, name="boosted"
    )
    flexible_double_pendulum(
        make_CosseratRod(interpolation="Quaternion", mixed=True, polynomial_degree=2),
        show_plots=False,
        name="default",
    )
