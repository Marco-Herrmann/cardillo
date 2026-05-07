import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.constraints import RigidConnection
from cardillo.forces import Force, B_Moment
from cardillo.math import e1, e2, e3, smoothstep2
from cardillo.rods import (
    RectangularCrossSection,
    CrossSectionInertias,
    Simo1986,
    animate_beam,
)
from cardillo.rods.cosseratRod import (
    make_CosseratRod,
)
from cardillo.solver import Newton, BackwardEuler, SolverOptions, ScipyDAE
from cardillo.visualization import Export

from cardillo_rods.beams import CosseratRodPG_SE3, CosseratRodPG_R12, CosseratRodPG_Quat

""" Cantilever oscillations """


def cantilever(
    Rod,  # TODO: add type hint
    *,
    nelements: int = 10,
    polynomial_degree: int = 2,
    #
    load_type: str = "moment_y",
    #
    VTK_export: bool = False,
):
    # geometry of the rod
    length = 1  # [m]
    # cross section properties
    width = 1e-3  # [m]
    height = width  # [m]
    density = 8e3  # [kg / m^3]

    # material properties
    E = 260.0e9  # [N / m^2]
    G = 100.0e9  # [N / m^2]

    # solver settings
    t1 = 10  # [s]
    dt = 1e-3  # [s]

    # # [m] -> [mm]
    # length *= 1e3

    # # cross section properties
    # width *= 1e3
    # height = width
    # density *= 1e-9

    # # [s] -> [ms]
    # t1 *= 1e3
    # dt *= 1e3

    # # [N / m^2] -> [kN / mm^2]
    # E *= 1e-9
    # G *= 1e-9

    cross_section = RectangularCrossSection(width, height)
    cross_section_inertias = CrossSectionInertias(density, cross_section)
    A = cross_section.area
    Ip, Iy, Iz = np.diagonal(cross_section.second_moment)

    # mass matrix terms
    print(f"M_r: density * A * length**3 / 3: {density * A * length**3 / 3}")
    print(f"M_psi: density * Iy * length**3 / 3: {density * Iy * length**3 / 3}")

    # rod stiffness properties
    # shear_corr = 5/6
    shear_corr = 1
    Ei = np.array([E * A, shear_corr * G * A, shear_corr * G * A])
    Fi = np.array([G * Ip, E * Iy, E * Iz])
    material_model = Simo1986(Ei, Fi)

    # construct system
    system = System()

    # compute straight initial configuration of cantilever
    q0, u0 = Rod.straight_initial_configuration(
        nelement=nelements,
        L=length,
        polynomial_degree=polynomial_degree,
    )

    # construct cantilever
    cantilever = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
        u0=u0,
        polynomial_degree=polynomial_degree,
        cross_section_inertias=cross_section_inertias,
    )

    clamping_left = RigidConnection(system.origin, cantilever, xi2=0)

    # assemble the system
    system.add(cantilever, clamping_left)

    # add load to predeform cantilever
    if load_type == "torsion":
        e = e1
        m_max = material_model.Fi[0] * np.pi / length
    elif load_type == "moment_y":
        e = e2
        m_max = material_model.Fi[1] * np.pi / length * 0.25
    elif load_type == "moment_z":
        e = e3
        m_max = material_model.Fi[2] * np.pi / length * 0.25

    if load_type in ["torsion", "moment_y", "moment_z"]:

        def M(t):
            return (
                e
                * m_max
                * (
                    smoothstep2(t, 0, 1)
                    # smoothstep2(t, 0, 0.25 * t1)
                    # - smoothstep2(t, 0.25 * t1, 0.3 * t1)
                )
            )

        # moment at cantilever tip
        load = B_Moment(M, cantilever, 1)
    elif load_type == "force":
        # spatially fixed load at cantilever tip
        P = lambda t: material_model.Fi[2] * (3 * t) / length**2 * 0.25
        F = lambda t: P(t) * e2
        load = Force(F, cantilever, 1)
    else:
        raise NotImplementedError

    system.add(load)
    system.assemble()

    # add Newton solver
    solver = Newton(
        system,
        n_load_steps=10,
    )

    # solve nonlinear static equilibrium equations
    sol = solver.solve()

    system.set_new_initial_state(q0=sol.q[-1], u0=sol.u[-1])

    system.remove(load)
    system.assemble()

    # solver = BackwardEuler(
    #     system,
    #     t1=t1,
    #     dt=dt,
    #     options=SolverOptions(newton_atol=1e-8),
    # )

    solver = ScipyDAE(
        system,
        t1=t1,
        dt=dt,
        # method="BDF",
        method="Radau",
        atol=1e-1,
        rtol=1e-1,
        stages=3,
    )

    sol = solver.solve()

    q = sol.q
    u = sol.u
    nt = len(q)
    t = sol.t[:nt]

    #################
    # post-processing
    #################

    # VTK export
    dir_name = Path(__file__).parent
    if VTK_export:
        system.export(dir_name, "vtk", sol, fps=500)

    # matplotlib visualization
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
        t,
        q,
        [cantilever],
        scale=length,
        scale_di=0.1,
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
        # Rod=make_CosseratRod(mixed=True, interpolation="SE3"),
        CosseratRodPG_SE3,
        # Rod=make_CosseratRod(mixed=False, interpolation="SE3"),
        # Rod=make_CosseratRod(mixed=True, interpolation="R12"),
        # Rod=make_CosseratRod(mixed=False, interpolation="R12"),
        # Rod=make_CosseratRod(mixed=False, interpolation="Quaternion"),
        # Rod=make_CosseratRod(mixed=True, interpolation="Quaternion"),
        nelements=10,
        polynomial_degree=3,
        # load_type="torsion",
        # load_type="moment_y",
        # load_type="moment_z",
        load_type="force",
        VTK_export=False,
    )
