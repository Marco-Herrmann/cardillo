"""Minimal debugging example for `Eigenmodes` / `FrequencyResponseFunction`.

A sphere hangs from a `TwoPointInteraction` + `KelvinVoigtElement` spring
("pendulum arm") anchored at the origin, with gravity pulling it down.
The static equilibrium is computed with `Newton`, then linearized
eigenmodes and the tip FRF (force -> tip displacement/rotation) are
evaluated about that equilibrium.

The `level` argument controls how much of the model is built, from the
purely analytic linear pendulum up to the full cable-on-ground model,
so the numerical model can be validated step by step against the
previous, simpler one:

    0: closed-form linear pendulum (e1, e2) + mass-spring-damper (e3);
       no `System`/solver involved at all.
    1: point mass (sphere) on the spring/gravity "pendulum", no cable.
    2: like 1, plus a slender cable clamped at xi=0 with the sphere
       rigidly attached at its tip (xi=1).
    3: like 2, plus a horizontal ground plane with frictionless
       (mu=0) `Sphere2Plane` contacts along the rod (and the tip
       sphere), so active/inactive `nla_N` is exercised in
       `Eigenmodes` and `FrequencyResponseFunction`.
    4: like 3, but with friction (mu=0.5), so `nla_F` is exercised
       as well.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cardillo import System
from cardillo.constraints import RigidConnection
from cardillo.constraints._base import ProjectedPositionOrientationBase
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

LEVEL_LABELS = {
    0: "linear pendulum",
    1: "mass only",
    2: "cable",
    3: "cable + ground (normal contact)",
    4: "cable + ground (frictional contact)",
}


class FixedOrientation(ProjectedPositionOrientationBase):
    """Locks a body's orientation to that of its reference subsystem
    while leaving its position completely free."""

    def __init__(
        self, subsystem1, subsystem2, xi1=None, xi2=None, name="fixed_orientation"
    ):
        super().__init__(
            subsystem1,
            subsystem2,
            constrained_axes_translation=(),
            projection_pairs_rotation=[(1, 2), (2, 0), (0, 1)],
            xi1=xi1,
            xi2=xi2,
            name=name,
        )


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


def _plot_frf(result, title):
    iom = result.iom
    fig, ax = plt.subplots(3, 3, sharex=True)
    labels_out = ["x", "y", "z"]
    labels_in = ["axial (e1)", "e2", "vertical (e3)"]
    for i in range(3):
        for j in range(3):
            # small floor avoids log-scale warnings on exactly-zero
            # (out-of-plane) entries, e.g. y-response for x-z loading
            ax[i, j].loglog(iom.imag, np.abs(result.frfs[:, i, j]))
            ax[i, j].grid()
            if i == 0:
                ax[i, j].set_title(f"F: {labels_in[j]}")
            if j == 0:
                ax[i, j].set_ylabel(f"tip {labels_out[i]}")
            if i == 2:
                ax[i, j].set_xlabel(r"$\omega$ [rad/s]")

    fig.suptitle(title)


def main(level, nelement=10, make_plot=True, blender_export=True, compute_frf=True):
    assert level in (0, 1, 2, 3, 4), f"level must be 0, 1, 2, 3 or 4, got {level}"
    print(f"level: {level}")

    g = 9.81  # gravity [m/s^2]

    #############################
    # sphere / "pendulum arm"
    #############################
    L = 10.0  # horizontal offset of the sphere [m]
    sphere_radius = 0.2
    sphere_density = 300  # [kg/m^3]
    sphere_mass = sphere_density * (4 / 3 * np.pi * sphere_radius**3)

    L0_spring = 3.0
    excentricity = 0.1
    KV_k = 1e3
    KV_d = 1e0

    iom = 1j * np.logspace(-1, 4, 2_000)

    if level == 0:
        ##################################################
        # closed-form linear pendulum / mass-spring model
        ##################################################
        frfs = np.zeros((len(iom), 6, 3), dtype=complex) * np.nan
        frfs[:, 0, 0] = frfs[:, 1, 1] = 1 / (iom**2 * L0_spring + g)
        frfs[:, 2, 2] = 1 / (iom**2 * sphere_mass + iom * KV_d + KV_k)

        result = FRFResult(
            frfs=frfs,
            iom=iom,
            L0_spring=L0_spring,
            sphere_mass=sphere_mass,
            KV_k=KV_k,
            KV_d=KV_d,
        )
        if make_plot:
            _plot_frf(result, f"Tip receptance FRF ({LEVEL_LABELS[level]})")
            plt.show()
        return result

    ##############
    # system setup
    ##############
    system = System()
    r_OP0_sphere = L * e1
    # spring attaches at the sphere's center, so the mounting point is
    # L0_spring away from there for l_ref=L0_spring to match the actual
    # initial spring length
    r_OP0_mounting = r_OP0_sphere + L0_spring * e3 + excentricity * e2

    #####################
    # sphere (Rigid Body)
    #####################
    sphere = Sphere(RigidBody)(
        radius=sphere_radius,
        subdivisions=2,
        q0=RigidBody.pose2q(r_OP0_sphere, np.eye(3)),
        density=sphere_density,
    )
    system.add(sphere)

    if level == 1:
        # a mass on a single point-to-point spring through its center
        # has no restoring torque at all, leaving the sphere's
        # orientation completely unconstrained (a singular rigid-body
        # spin mode); lock its orientation instead, since the cable
        # (level >= 2) would otherwise do this via `connection` below
        orientation_lock = FixedOrientation(
            sphere, system.origin, name="sphere_orientation_lock"
        )
        system.add(orientation_lock)

    # force element ("pendulum arm")
    interaction = TwoPointInteraction(sphere, system.origin, B_r_CP2=r_OP0_mounting)
    KV_element = KelvinVoigtElement(
        interaction, KV_k, KV_d, l_ref=L0_spring, compliance_form=True
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

    # gravity acting on the sphere
    gravity_sphere = Force(
        lambda t: np.array([0.0, 0.0, -sphere_mass * g]) * (1.0 if level == 1 else t),
        sphere,
        name="gravity_sphere",
    )
    system.add(gravity_sphere)

    ###########
    # add cable
    ###########
    if level >= 2:
        #####################
        # geometry & material
        #####################
        radius = 5e-3  # cable radius [m]

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
        b = lambda t, xi: t * np.array([0.0, 0.0, -g * A * density])

        Cable = make_CosseratRod(polynomial_degree=2)

        Q = Cable.straight_configuration(nelement, L - sphere_radius)
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

        # rigidly connect the cable's tip to the center of the sphere
        connection = RigidConnection(
            rod, sphere, xi1=1, r_OJ0=r_OP0_sphere, name="cable_to_sphere"
        )
        system.add(connection)

    ########
    # ground
    ########
    if level >= 3:
        z_ground = -0.7
        dimensions = np.array([L + 2, 2, 1])
        r_OP_frame = np.array([L / 2, 0.0, z_ground - dimensions[2] / 2])
        ground = Box(Frame)(dimensions=dimensions, r_OP=r_OP_frame, name="ground")
        system.add(ground)

        mu = 0.0 if level == 3 else 0.5

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

    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=True))

    #########
    # statics
    #########
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

    if blender_export:
        dir_name = Path(__file__).parent
        suffix = f"_level{level}"
        system.export_blender(
            dir_name, f"blender_static{suffix}", sol, create_blend=True
        )
        system.export_blender(
            dir_name, f"blender_eigenmodes{suffix}", sol_eig, create_blend=True
        )

    if not compute_frf:
        return True

    ############################
    # frequency response function
    ############################
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

    if make_plot:
        _plot_frf(result, f"Tip receptance FRF ({LEVEL_LABELS[level]})")
        plt.show()

    return result


if __name__ == "__main__":
    # build up the model level by level
    levels = np.arange(5)

    # make blender files with 40 elements
    {
        level: main(level=level, nelement=40, make_plot=False, compute_frf=False)
        for level in levels
    }

    # compare the tip FRF
    results = {
        level: main(level=level, nelement=10, make_plot=False) for level in levels
    }
    iom = results[0].iom

    fig, ax = plt.subplots(3, 3, sharex=True)
    labels_out = ["x", "y", "z"]
    labels_in = ["axial (e1)", "e2", "vertical (e3)"]
    for i in range(3):
        for j in range(3):
            for level, result in results.items():
                # small floor avoids log-scale warnings on exactly-zero
                # (out-of-plane) entries, e.g. y-response for x-z loading
                ax[i, j].loglog(
                    iom.imag,
                    np.abs(result.frfs[:, i, j]),
                    label=LEVEL_LABELS[level],
                )
            ax[i, j].grid()
            if i == 0:
                ax[i, j].set_title(f"F: {labels_in[j]}")
            if j == 0:
                ax[i, j].set_ylabel(f"tip {labels_out[i]}")
            if i == 2:
                ax[i, j].set_xlabel(r"$\omega$ [rad/s]")
    ax[2, 2].legend()
    fig.suptitle("Tip receptance FRF: comparison across levels")
    plt.show()
