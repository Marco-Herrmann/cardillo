import matplotlib.pyplot as plt
import numpy as np

from cardillo import System
from cardillo.constraints import RigidConnection
from cardillo.forces import Force, B_Moment
from cardillo.math import e2, e3
from cardillo.rods import animate_beam
from cardillo.rods_new import (
    RectangularCrossSection,
    CrossSectionInertias,
    make_CosseratRod,
    Simo1986,
    Material3D,
    Elasticity,
    Rectangle_Quadrature,
)
from cardillo.solver import MoreauTheta, SolverOptions
from cardillo.utility.sensor import Sensor


def cantilever(
    Rod,
    *,
    nelements: int = 10,
    t1: float = 0.5,
    dt: float = 1e-3,
    theta: float = 0.6,
    t_ramp: float = 0.1,
    show_plots: bool = False,
):
    # geometry (units: mm, N, MPa, t, s)
    length = 1000
    width = 20
    cross_section = RectangularCrossSection(width, width)
    I1, I2, I3 = np.diag(cross_section.second_moment(0.0))
    A = cross_section.area(0.0)

    # material model
    E = 210e3
    nu = 0.3
    density = 7.85e-9
    cross_section_inertias = CrossSectionInertias(density, cross_section)
    cross_section_quadrature = Rectangle_Quadrature(5, 5, width, width)
    material_model_3D = Elasticity(E, nu)
    material_model1 = Material3D(material_model_3D, cross_section_quadrature)

    # comparison: Simo1986 with the same cross-section stiffnesses
    G = E / (2 * (1 + nu))
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * I1, E * I2, E * I3])
    material_model2 = Simo1986(Ei, Fi)

    # in-plane tip loads (x-y plane), linearly ramped up and then held
    EI = E * I3
    F_max = 1.0 * EI / length**2
    M_max = 0.5 * EI / length
    load_factor = lambda t: min(t / t_ramp, 1.0)
    F = lambda t: -load_factor(t) * F_max * e2
    M = lambda t: load_factor(t) * M_max * e3

    # initialize system
    system = System()

    # rods along e1, the second one shifted in e2-direction
    rods = []
    q0s = []
    for i, (material_model, name) in enumerate(
        [(material_model1, "Material3D"), (material_model2, "Simo1986")]
    ):
        q0 = Rod.straight_configuration(nelements, length, r_OP0=i * 0.2 * length * e2)
        rod = Rod(
            cross_section,
            material_model,
            nelements,
            Q=q0,
            cross_section_inertias=cross_section_inertias,
            name=f"rod_{name}",
        )
        clamping = RigidConnection(rod, system.origin, xi1=0, name=f"clamping_{name}")
        force = Force(F, rod, xi=1, name=f"tip_force_{name}")
        moment = B_Moment(M, rod, xi=1, name=f"tip_moment_{name}")
        sensor = Sensor(rod, xi=1, name=f"tip_{name}")
        system.add(rod, clamping, force, moment, sensor)
        rods.append(rod)
        q0s.append(q0)

    # assemble system
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    ############
    # simulation
    ############
    # theta = 0.5 becomes unstable (also with Simo1986), theta > 0.5 adds damping
    sol = MoreauTheta(system, t1, dt, theta=theta).solve()

    #################
    # post-processing
    #################
    t = sol.t
    q = sol.q

    # tip displacement
    fig, ax = plt.subplots(1, 1)
    fig.suptitle("Tip displacement")
    for rod, q0, ls in zip(rods, q0s, ["-", "--"]):
        qDOF_tip = rod.local_qDOF_P(1)
        r_OP0_tip = rod.r_OP(0, q0[qDOF_tip], 1)
        delta_tip = np.array(
            [
                rod.r_OP(ti, qi[rod.qDOF][qDOF_tip], 1) - r_OP0_tip
                for ti, qi in zip(t, q)
            ]
        )
        name = rod.name.removeprefix("rod_")
        ax.plot(t, delta_tip[:, 0], f"{ls}r", label=rf"$\Delta x$ {name}")
        ax.plot(t, delta_tip[:, 1], f"{ls}g", label=rf"$\Delta y$ {name}")
        ax.plot(t, delta_tip[:, 2], f"{ls}b", label=rf"$\Delta z$ {name}")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("tip displacement")
    ax.grid()
    ax.legend()

    if show_plots:
        fig_animate, ax_animate, anim = animate_beam(
            t,
            q,
            rods,
            scale=length,
            scale_di=0.05 * length,
            show=False,
            n_frames=len(t) // 5,
            repeat=True,
        )
        ax_animate.view_init(elev=90, azim=-90)
        plt.show()


if __name__ == "__main__":
    Rod = make_CosseratRod(
        polynomial_degree=2,
        # idx_displacement_based=np.arange(6),
    )

    cantilever(
        Rod,
        nelements=10,
        t1=0.5,
        dt=1e-3,
        t_ramp=0.1,
        show_plots=True,
    )
