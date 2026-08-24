import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import warnings

from cardillo import System
from cardillo.constraints import RigidConnection, Prismatic
from cardillo.constraints._base import ProjectedPositionOrientationBase
from cardillo.forces import Force, B_Force, Moment, B_Moment
from cardillo.math import e1, e2, e3, A_IB_basic

from cardillo.rods_new import (
    Simo1986,
    CircularCrossSection,
    RectangularCrossSection,
    CrossSectionInertias,
)
from cardillo.rods_new2 import make_CosseratRod
from cardillo.solver import Newton, SolverOptions, Eigenmodes
from cardillo.utility.sensor import Sensor, SensorRecords


def string(
    Rod,
    *,
    nelements: int = 10,
    reversal: bool = False,
    #
    n_load_steps: int = 5,
):

    # geometry
    L = 0.648
    r = 0.127e-3

    cross_section = CircularCrossSection(r)
    cross_section_export = CircularCrossSection(50 * r)
    A = cross_section.area(0.0)
    Ip, Iy, Iz = np.diag(cross_section.second_moment(0.0))

    # material
    rho = 7_850
    E = 210e9
    G = 80.8e9

    # build model
    q0 = Rod.straight_configuration(nelements, L)
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * Iy, E * Iz])
    material_model = Simo1986(Ei, Fi)
    cross_section_inertias = CrossSectionInertias(rho, cross_section)
    cross_section_inertias = CrossSectionInertias(
        A_rho0=rho * A, B_I_rho0=np.diag([Ip * rho, 0.0, 0.0])
    )

    # create system
    system = System()

    # rod
    rod = Rod(
        cross_section_export,
        material_model,
        nelements,
        Q=q0,
        cross_section_inertias=cross_section_inertias,
        name="string",
    )
    system.add(rod)

    # pretension
    F0 = 72.0
    # pretension = Force(lambda t: t * F0 * e3 / 5e3, rod, xi=0.5, name="pretension")
    pretension = Force(lambda t: t * F0 * e1, rod, xi=1.0, name="pretension")
    system.add(pretension)

    # constraints
    # constraints = [
    #     RigidConnection(rod, system.origin, xi1=0),
    #     Prismatic(rod, system.origin, axis=0, xi1=1),
    # ]
    constraints = [
        ProjectedPositionOrientationBase(
            rod, system.origin, [0, 1, 2], [(1, 2)], xi1=0
        ),
        ProjectedPositionOrientationBase(rod, system.origin, [1, 2], [], xi1=1),
    ]
    system.add(*constraints)

    # assemble system
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    # solve
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        options=SolverOptions(newton_max_iter=50),
    )
    sol = solver.solve()

    # eigenmodes
    solver_eigenmodes = Eigenmodes(system, sol)
    sol_eigenmodes = solver_eigenmodes.solve(-1)

    #################
    # post-processing
    #################
    # blender-export
    dir_name = Path(sys.argv[0]).parent
    system.export_blender(dir_name, f"blender", sol, create_blend=True)
    system.export_blender(dir_name, f"blender_eig", sol_eigenmodes, create_blend=True)

    # theoretical value
    f0_analytical = 1 / (2 * r * L) * np.sqrt(F0 / (np.pi * rho))
    omega0_analytical = 2 * np.pi * f0_analytical
    print(omega0_analytical)
    print(sol_eigenmodes.omegas[0])


if __name__ == "__main__":
    Rod = make_CosseratRod(
        polynomial_degree=1,
        # quadrature_int=(2, "Gauss"),
        idx_constraints=[0, 1, 2],
        # idx_displacement_based=[0, 1, 2],
        # idx_displacement_based=[4, 5],
        # idx_displacement_based=[0, 1, 2, 3, 4, 5],
    )
    string(Rod)
