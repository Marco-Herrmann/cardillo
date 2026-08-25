"""Minimal debugging example for `Eigenmodes` / `FrequencyResponseFunction`.

A slender, straight cable is clamped at xi=0 and pulled at xi=1 with a
force that is mostly axial with a small upward component. Gravity acts
as a distributed load along the whole rod. The static equilibrium is
computed with `Newton`, then linearized eigenmodes and the tip FRF
(force -> tip displacement/rotation) are evaluated about that
equilibrium.

This system was intentionally kept free of unilateral/frictional
contacts so it exercises the "clean" path of `Eigenmodes` and
`FrequencyResponseFunction` (no active `nla_N`/`nla_F`). A ground
plane with `Sphere2Plane` contacts can be added later along the rod to
also exercise the contact-related code paths (see `WITH_GROUND`
below).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cardillo import System
from cardillo.constraints import RigidConnection
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

# toggle for a future extension: add a ground plane with Sphere2Plane
# contacts along the rod (see cardillo.contacts.Sphere2Plane) to also
# exercise the frictional-contact code paths of the FRF solver.
WITH_GROUND = False


if __name__ == "__main__":
    #####################
    # geometry & material
    #####################
    L = 10.0  # cable length [m]
    radius = 5e-3  # cable radius [m]
    nelement = 10

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
    b = lambda t, xi: np.array([0.0, 0.0, -g * A * density * t])

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

    # tip force: mostly axial (e1) with a slight upward (e3) component
    F_axial = 1.0e3  # [N]
    F_axial = 2.0e1  # [N]
    F_up = 0.5 * g * A * density * L  # [Nd]
    print(f"{F_axial = }, {F_up = }")
    F_perp = F_up * 0.05  # [N]
    F = lambda t: t * (F_axial * e1 + F_perp * e2 + F_up * e3)
    force = Force(F, rod, xi=1)
    system.add(force)

    # sensor at the tip; together with `force` above this defines the
    # 6 (translation + rotation) x 3 (force) input/output pair used by
    # the FRF solver
    sensor = Sensor(rod, xi=1, name="Tip")
    system.add(sensor)

    if WITH_GROUND:
        raise NotImplementedError(
            "ground contact (Sphere2Plane) is not wired up yet in this example"
        )

    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    ############
    # statics
    ############
    n_load_steps = 10
    solver = Newton(
        system, n_load_steps=n_load_steps, options=SolverOptions(newton_atol=1e-10)
    )
    sol = solver.solve()

    r_OP_tip = rod.r_OP(sol.t[-1], sol.q[-1][sensor.qDOF], 1)
    print(f"tip position: {r_OP_tip} (undeformed: {L * e1})")

    ##############
    # eigenmodes
    ##############
    solver_eig = Eigenmodes(system, sol)
    sol_eig = solver_eig.solve(-1)
    print(f"first 10 natural frequencies [rad/s]:\n{sol_eig.omegas[:10]}")

    dir_name = Path(__file__).parent
    system.export_blender(dir_name, "blender_static", sol, create_blend=True)
    system.export_blender(dir_name, "blender_eigenmodes", sol_eig, create_blend=True)

    ############################
    # frequency response function
    ############################
    iom = 1j * np.logspace(-1, 4, 200)
    solver_frf = FrequencyResponseFunction(system, sol)
    frfs = solver_frf.solve(-1, iom)  # shape (len(iom), 6, 3)

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
    fig.suptitle("Tip receptance FRF of clamped cable")
    plt.show()
