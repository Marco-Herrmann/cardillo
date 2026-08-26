"""Minimal debugging example for `Eigenmodes` / `FrequencyResponseFunction`.

A slender, straight cable is clamped at xi=0 and pulled at xi=1 with a
force that is mostly axial with a small upward component. Gravity acts
as a distributed load along the whole rod. The static equilibrium is
computed with `Newton`, then linearized eigenmodes and the tip FRF
(force -> tip displacement/rotation) are evaluated about that
equilibrium.

With `WITH_GROUND = True` a horizontal ground plane is added below the
cable with `Sphere2Plane` contacts along the rod (and the tip sphere),
so active/inactive `nla_N`/`nla_F` are exercised in `Eigenmodes` and
`FrequencyResponseFunction`. With `WITH_GROUND = False` the system
stays free of unilateral/frictional contacts, exercising the "clean"
path instead.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cardillo import System
from cardillo.constraints import RigidConnection
from cardillo.contacts import Sphere2Plane
from cardillo.discrete import Box, Sphere, RigidBody, Frame
from cardillo.force_laws import KelvinVoigtElement
from cardillo.interactions import TwoPointInteraction
from cardillo.forces import Force
from cardillo.math import e1, e2, e3
from cardillo.rods_new import (
    CircularCrossSection,
    CrossSectionInertias,
    Simo1986,
    make_CosseratRod,
)
from cardillo.solver import (
    Newton,
    SolverOptions,
    Eigenmodes,
    FrequencyResponseFunction,
)
from cardillo.utility.sensor import Sensor


@dataclass
class FRFResult:
    """Bundles the tip FRF together with the parameters needed to
    plot analytic reference curves against it (see `__main__` below)."""

    frfs: np.ndarray
    iom: np.ndarray
    L0_spring: float
    sphere_mass: float
    KV_k: float
    KV_d: float


def main(with_ground, make_plot=True, blender_export=True):
    #####################
    # geometry & material
    #####################
    L = 10.0  # cable length [m]
    radius = 5e-3  # cable radius [m]
    nelement = 50

    E = 2.0e11  # Young's modulus [Pa] (steel)
    G = 8.0e10  # shear modulus [Pa]
    density = 7.8e3  # [kg/m^3]

    cross_section = CircularCrossSection(radius)
    A = cross_section.area(0.0)
    I1, I2, I3 = np.diag(cross_section.second_moment(0.0))

    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * I1, E * I2, E * I3])
    material_model = Simo1986(Ei, Fi)

    cross_section_inertias = CrossSectionInertias(density, cross_section)

    # gravity as distributed load along the rod
    g = 9.81
    b = lambda t, xi: t * np.array([0.0, 0.0, -g * A * density])

    ##############
    # system setup
    ##############
    system = System()

    Cable = make_CosseratRod(polynomial_degree=2)

    Q = Cable.straight_configuration(nelement, L)
    rod = Cable(
        cross_section,
        material_model,
        nelement,
        Q=Q,
        q0=Q,
        cross_section_inertias=cross_section_inertias,
        distributed_load=[b, None],
        name="cable",
    )
    system.add(rod)

    # clamp at xi=0
    clamping = RigidConnection(rod, system.origin, xi1=0)
    system.add(clamping)

    #######################
    # sphere at cable's tip
    #######################
    sphere_radius = 0.2
    sphere_density = 300  # [kg/m^3]
    sphere_mass = sphere_density * (4 / 3 * np.pi * sphere_radius**3)

    r_OP_tip0 = L * e1
    r_OP0_sphere = r_OP_tip0 + sphere_radius * e1
    sphere = Sphere(RigidBody)(
        radius=sphere_radius,
        subdivisions=2,
        q0=RigidBody.pose2q(r_OP0_sphere, np.eye(3)),
        density=sphere_density,
    )
    system.add(sphere)

    # rigidly connect the cable's tip center of the sphere
    connection = RigidConnection(
        rod, sphere, xi1=1, r_OJ0=r_OP0_sphere, name="cable_to_sphere"
    )
    system.add(connection)

    # gravity acting on the sphere
    gravity_sphere = Force(
        lambda t: t * np.array([0.0, 0.0, -sphere_mass * g]),
        sphere,
        name="gravity_sphere",
    )
    system.add(gravity_sphere)

    # force element
    L0_spring = 3.0
    excentricity = 0.1
    interaction = TwoPointInteraction(
        sphere, system.origin, B_r_CP2=r_OP0_sphere + L0_spring * e3 + excentricity * e2
    )
    KV_k = 1e3
    KV_d = 1e0
    KV_element = KelvinVoigtElement(
        interaction, KV_k, KV_d, l_ref=L0_spring, compliance_form=False
    )
    system.add(interaction, KV_element)

    # "control input force"
    F = lambda t: np.zeros(3)
    force = Force(F, sphere, name="tip_force")
    system.add(force)

    # sensor at the tip; together with `force` above
    # this defines the 6 (translation + rotation) x 3 (force)
    # input/output pair used by the FRF solver
    sensor = Sensor(sphere, name="Tip")
    system.add(sensor)

    if with_ground:
        ####################
        # ground with contacts
        ####################
        z_ground = -0.7
        dimensions = np.array([L + 2, 2, 1])
        r_OP_frame = np.array([L / 2, 0.0, z_ground - dimensions[2] / 2])
        ground = Box(Frame)(dimensions=dimensions, r_OP=r_OP_frame, name="ground")
        system.add(ground)

        mu = 0.5

        # point contacts along the rod (skip node 0, it is clamped and
        # never moves, so a contact there would be permanently inert)
        for node in range(1, rod.nnodes):
            contact_rod = Sphere2Plane(
                ground,
                rod,
                mu=mu,
                radius=radius,
                B_r_CP1=np.array([0.0, 0.0, dimensions[2] / 2]),
                xi2=node / (rod.nnodes - 1),
                name=f"contact_ground_rod_{node:02d}",
            )
            system.add(contact_rod)

        # the tip sphere itself, using its actual radius
        contact_sphere = Sphere2Plane(
            ground, sphere, mu=mu, radius=sphere_radius, name="contact_ground_sphere"
        )
        system.add(contact_sphere)

    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    ############
    # statics
    ############
    n_load_steps = 10
    solver = Newton(
        system, n_load_steps=n_load_steps, options=SolverOptions(newton_atol=1e-10)
    )
    sol = solver.solve()

    r_OP_tip = sphere.r_OP(sol.t[-1], sol.q[-1][sensor.qDOF])
    print(f"tip position: {r_OP_tip} (undeformed: {r_OP0_sphere})")

    ##############
    # eigenmodes
    ##############
    solver_eig = Eigenmodes(system, sol)
    sol_eig = solver_eig.solve(-1)
    print(f"first 10 natural frequencies [rad/s]:\n{sol_eig.omegas[:10]}")

    dir_name = Path(__file__).parent
    if with_ground and blender_export:
        system.export_blender(dir_name, "blender_static_ground", sol, create_blend=True)
        system.export_blender(
            dir_name, "blender_eigenmodes_ground", sol_eig, create_blend=True
        )
    elif blender_export:
        system.export_blender(dir_name, "blender_static", sol, create_blend=True)
        system.export_blender(
            dir_name, "blender_eigenmodes", sol_eig, create_blend=True
        )

    ############################
    # frequency response function
    ############################
    iom = 1j * np.logspace(-1, 4, 2_000)
    solver_frf = FrequencyResponseFunction(system, sol)
    frfs = solver_frf.solve(-1, iom)  # shape (len(iom), 6, 3)

    result = FRFResult(
        frfs=frfs,
        iom=iom,
        L0_spring=L0_spring,
        sphere_mass=sphere_mass,
        KV_k=KV_k,
        KV_d=KV_d,
    )

    if not make_plot:
        return result

    fig, ax = plt.subplots(3, 3, sharex=True)
    labels_out = ["x", "y", "z"]
    labels_in = ["axial (e1)", "e2", "vertical (e3)"]
    for i in range(3):
        for j in range(3):
            # small floor avoids log-scale warnings on exactly-zero
            # (out-of-plane) entries, e.g. y-response for x-z loading
            ax[i, j].loglog(iom.imag, np.abs(frfs[:, i, j]) + 1e-30)
            ax[i, j].grid()
            if i == 0:
                ax[i, j].set_title(f"F: {labels_in[j]}")
            if j == 0:
                ax[i, j].set_ylabel(f"tip {labels_out[i]}")
            if i == 2:
                ax[i, j].set_xlabel(r"$\omega$ [rad/s]")

    # pure pendulum (e1, e2) and mass-spring-oszillator with sqrt(g/l) (e3)
    ax[0, 0].loglog(iom.imag, np.abs(1 / (iom**2 * L0_spring + 9.81)), "--")
    ax[1, 1].loglog(iom.imag, np.abs(1 / (iom**2 * L0_spring + 9.81)), "--")
    ax[2, 2].loglog(
        iom.imag, np.abs(1 / (iom**2 * sphere_mass + iom * KV_d + KV_k)), "--"
    )
    ax[2, 2].loglog(
        iom.imag, np.abs(1 / (iom**2 * sphere_mass + iom * 0.0 + KV_k)), "--"
    )
    fig.suptitle("Tip receptance FRF of clamped cable")
    plt.show()

    return result


if __name__ == "__main__":
    # main(with_ground=True)
    # exit()

    # compare
    result_no_ground = main(with_ground=False, blender_export=True, make_plot=False)
    result_ground = main(with_ground=True, blender_export=True, make_plot=False)

    # iom and the analytic-reference parameters are identical for both
    # runs, so either result carries what's needed for the plot below
    iom = result_no_ground.iom
    L0_spring = result_no_ground.L0_spring
    sphere_mass = result_no_ground.sphere_mass
    KV_k = result_no_ground.KV_k
    KV_d = result_no_ground.KV_d

    fig, ax = plt.subplots(3, 3, sharex=True)
    labels_out = ["x", "y", "z"]
    labels_in = ["axial (e1)", "e2", "vertical (e3)"]
    for i in range(3):
        for j in range(3):
            # small floor avoids log-scale warnings on exactly-zero
            # (out-of-plane) entries, e.g. y-response for x-z loading
            ax[i, j].loglog(iom.imag, np.abs(result_no_ground.frfs[:, i, j]) + 1e-30)
            ax[i, j].loglog(iom.imag, np.abs(result_ground.frfs[:, i, j]) + 1e-30)
            ax[i, j].grid()
            if i == 0:
                ax[i, j].set_title(f"F: {labels_in[j]}")
            if j == 0:
                ax[i, j].set_ylabel(f"tip {labels_out[i]}")
            if i == 2:
                ax[i, j].set_xlabel(r"$\omega$ [rad/s]")

    # pure pendulum (e1, e2) and mass-spring-oszillator with sqrt(g/l) (e3)
    H_pendulum = np.abs(1 / (iom**2 * L0_spring + 9.81))
    H_osc_damped = np.abs(1 / (iom**2 * sphere_mass + iom * KV_d + KV_k))
    H_osc_undamped = np.abs(1 / (iom**2 * sphere_mass + iom * 0.0 + KV_k))
    ax[0, 0].loglog(iom.imag, H_pendulum, "--")
    ax[1, 1].loglog(iom.imag, H_pendulum, "--")
    ax[2, 2].loglog(iom.imag, H_osc_damped, "--")
    ax[2, 2].loglog(iom.imag, H_osc_undamped, "--")
    ax[2, 2].legend(["No ground", "Ground", "Pendulum", "Unpdamed Pendulum"])
    fig.suptitle("Tip receptance FRF of clamped cable")
    plt.show()
