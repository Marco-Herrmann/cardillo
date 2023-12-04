from cardillo.rods import (
    RectangularCrossSection,
    CrossSectionInertias,
    Simo1986,
    animate_beam,
)

from cardillo.rods.cosseratRod import (
    make_CosseratRod_SE3,
    make_CosseratRod_R12,
    make_CosseratRod_Quat,
)

from cardillo.constraints import RigidConnection
from cardillo.solver import Newton, BackwardEuler, SolverOptions
from cardillo.forces import Force, K_Moment, Moment, K_Force

from cardillo.math import e1, e2, e3, smoothstep0

from cardillo.visualization import Export

from cardillo import System

import numpy as np
from math import pi

import matplotlib.pyplot as plt
from pathlib import Path

""" Cantilever oscillations """


def cantilever(
    Rod,
    nelements=10,
    polynomial_degree=2,
    n_load_steps=10,
    load_type="force",
    VTK_export=False,
    reduced_integration=True,
):
    # geometry of the rod
    length = 2 * np.pi

    # cross section properties for visualization purposes
    slenderness = 1.0e2
    width = length / slenderness
    density = 80  # [kg / m^3]
    # cross_section = RectangularCrossSection(density, width, width)
    cross_section = RectangularCrossSection(width, 3 * width)
    cross_section_inertias = CrossSectionInertias(density, cross_section)

    A = cross_section.area
    Ip, Iy, Iz = np.diagonal(cross_section.second_moment)

    E = 210.0e6
    G = 80.0e6
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * Iy, E * Iz])

    material_model = Simo1986(Ei, Fi)

    # construct system
    system = System()

    # compute straight initial configuration of cantilever
    q0 = Rod.straight_configuration(
        nelements, length, polynomial_degree=polynomial_degree
    )
    # construct cantilever
    cantilever = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
        polynomial_degree=polynomial_degree,
        reduced_integration=reduced_integration,
    )

    clamping_left = RigidConnection(system.origin, cantilever, frame_ID2=(0,))

    # assemble the system
    system.add(cantilever)
    system.add(clamping_left)

    if load_type == "force":
        # spatially fixed load at cantilever tip
        P = lambda t: 5 * t * material_model.Fi[2] / length**2
        F = lambda t: -P(t) * e2
        load = K_Force(F, cantilever, (1,))

    elif load_type == "torsion":
        # spatially fixed load at cantilever tip
        n = 1
        P = lambda t: material_model.Fi[0] * t * 2 * pi * n / length
        F = lambda t: P(t) * e1
        load = K_Moment(F, cantilever, (1,))

    elif load_type == "force_torsion":
        # spatially fixed load at cantilever tip
        P = (
            lambda t: 5
            * smoothstep0(t, 0, 0.5)
            * t
            * material_model.Fi[2]
            / length**2
        )
        F = lambda t: -P(t) * e2
        load = Force(F, cantilever, (1,))

        # n = 0
        # P = lambda t: material_model.Fi[0] * smoothstep0(t, 0.8, 1) * t * 2 * pi * n / length
        # F = lambda t: P(t) * e1
        # load1 = K_Moment(F, cantilever, (1,))
        # system.add(load1)

    else:
        raise NotImplementedError("This load type has not been implemented")

    system.add(load)
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    # add Newton solver
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
    )

    # solve nonlinear static equilibrium equations
    sol = solver.solve()

    # # extract solutions
    # q = sol.q
    # nt = len(q)
    # t = sol.t[:nt]

    system_ph1 = system.deepcopy(
        sol, options=SolverOptions(compute_consistent_initial_conditions=False)
    )
    system.remove(load)
    # system.remove(load1)
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    # dt = 1e-2
    # dt = 1e-2
    dt = 1e-2
    solver = BackwardEuler(
        system,
        t1=2,
        dt=dt,
        options=SolverOptions(newton_atol=1e-8),
    )

    sol = solver.solve()

    q = sol.q
    nt = len(q)
    t = sol.t[:nt]

    # VTK export
    if VTK_export:
        path = Path(__file__)
        e = Export(path.parent, path.stem, True, 30, sol)
        e.export_contr(
            cantilever,
            level="centerline + directors",
            num=3 * nelements,
            file_name="cantilever_curve",
        )
        e.export_contr(
            cantilever,
            continuity="C0",
            level="volume",
            n_segments=nelements,
            num=3 * nelements,
            file_name="cantilever_volume",
        )

    # matplotlib visualization
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
        t,
        q,
        [cantilever],
        scale=length,
        scale_di=0.5,
        show=False,
        n_frames=cantilever.nelement + 1,
        repeat=True,
    )

    # add plane with z-direction as normal
    X_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    Y_z = np.linspace(-1.1 * length, 1.1 * length, num=2)
    X_z, Y_z = np.meshgrid(X_z, Y_z)
    Z_z = np.zeros_like(X_z)
    ax1.plot_surface(X_z, Y_z, Z_z, alpha=0.2)

    # plot animation
    ax1.azim = -90
    ax1.elev = 72

    plt.show()


if __name__ == "__main__":
    # SE3 interpolation:
    cantilever(
        Rod=make_CosseratRod_SE3(mixed=True, constraints=[1, 2]),
        nelements=5,
        polynomial_degree=1,
        n_load_steps=10,
        load_type="force",
        # load_type="force_torsion",
        VTK_export=False,
    )

    # cantilever(
    #     Rod=make_CosseratRod_SE3(mixed=False),
    #     nelements=5,
    #     polynomial_degree=1,
    #     n_load_steps=3,
    #       load_type="force",
    #     VTK_export=True
    # )
