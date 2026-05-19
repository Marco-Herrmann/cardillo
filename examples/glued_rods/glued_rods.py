from math import pi
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.constraints import RigidConnection
from cardillo.forces import B_Moment, Force
from cardillo.math import e1, e2, e3
from cardillo.rods import (
    CircularCrossSection,
    RectangularCrossSection,
    Simo1986,
)
from cardillo.rods.cosseratRod import (
    make_CosseratRod,
)
from cardillo.solver import Newton, SolverOptions


def glued_rods(
    Rod,  # TODO: add type hint
    constitutive_law=Simo1986,  # TODO: add type hint
    *,
    nelements: int = 10,
    #
    n_load_steps: int = 10,
    #
    VTK_export: bool = False,
    name: str = "simulation",
):
    # handle name
    plot_name = name.replace("_", " ")
    save_name = name.replace(" ", "_")

    ############
    # parameters
    ############
    # geometry of the rod
    length = 1.0

    # cross section properties only for visualization purposes
    width = length / 5
    height = length / 10
    cross_section_rect = RectangularCrossSection(width, height)

    # material properties
    E = 1e5
    mu = 0.3
    G = E / (2 * (1 + mu))
    A = cross_section_rect.area
    Ip, Iy, Iz = np.diag(cross_section_rect.second_moment)
    Ei = np.array([E*A, G*A, G*A])
    Fi = np.array([G * Ip, E*Iy, E*Iz])

    material_model = constitutive_law(Ei, Fi)

    # initialize system
    system = System()

    #####
    # rod
    #####
    # compute straight initial configuration of cantilever
    Q0_a = Rod.straight_configuration(nelements, length)
    # construct cantilever
    rod_a = Rod(
        cross_section_rect,
        material_model,
        nelements,
        Q=Q0_a,
        q0=Q0_a,
        name="rodA"
    )

    ##########
    # clamping
    ##########
    clamping = RigidConnection(system.origin, rod_a, xi2=0)
    system.add(rod_a, clamping)

    ###########################
    # apply pretension on rod a
    ###########################
    length_tensioned = 1.5 * length
    F = e1 * material_model.Ei[0] * (length_tensioned - length) / length
    force_a = Force(lambda t: t * F, rod_a, 1.0)
    force_b = Force(lambda t: (1.0 - t) * F, rod_a, 1.0)
    system.add(force_a)

    # assemble system
    system.assemble()

    ############
    # simulation
    ############
    solver_a = Newton(system, n_load_steps=n_load_steps)  # create solver
    sol_a = solver_a.solve()  # solve static equilibrium equations

    #################
    # post-processing
    #################
    # VTK export
    dir_name = Path(__file__).parent
    if VTK_export:
        # export only nodal quantities for fast export (circle)
        # rod_a._export_dict["level"] = "NodalVolume"
        rod_a._export_dict["stresses"] = True

        # export
        system.export(dir_name, f"vtk/{save_name}/sol_a", sol_a)

    #######
    # rod b
    #######
    # compute straight initial configuration of cantilever
    Q0_b = Rod.straight_configuration(nelements, length_tensioned, r_OP0=np.array([0.0, width, 0.0]))
    # construct cantilever
    rod_b = Rod(
        cross_section_rect,
        material_model,
        nelements,
        Q=Q0_b,
        q0=Q0_b,
        name="rodB"
    )
    
    ################################
    # connect rods discrete at nodes
    ################################
    nnodes = rod_a.nnodes_r
    xis_connect = np.linspace(0, 1, nnodes)
    constraints = []
    for node in range(nnodes):
        xi_node = xis_connect[node]
        constraints.append(
            RigidConnection(rod_a, rod_b, xi1=xi_node, xi2=xi_node, name=f"rigid_connection_{node}")
        )

    ###############################
    # change external load at right
    ###############################
    # TODO: add external moment instead of force at the right side
    M = width * F


    ###############
    # update system
    ###############
    system.set_new_initial_state(sol_a.q[-1], sol_a.u[-1], 0.0)
    system.remove(force_a)
    system.add(force_b, rod_b, *constraints)

    # assemble system
    system.assemble()

    ############
    # simulation
    ############
    solver_b = Newton(system, n_load_steps=n_load_steps)  # create solver
    sol_b = solver_b.solve()  # solve static equilibrium equations

    #################
    # post-processing
    #################
    # VTK export
    dir_name = Path(__file__).parent
    if VTK_export:
        # export only nodal quantities for fast export (circle)
        # rod_b._export_dict["level"] = "NodalVolume"
        rod_b._export_dict["stresses"] = True
        
        rod_a._export_dict["stresses"] = True

        # export
        system.export(dir_name, f"vtk/{save_name}/sol_b", sol_b)

if __name__ == "__main__":
    glued_rods(
        # Rod=make_CosseratRod(interpolation="SE3", mixed=True, constraints=[0, 1, 2]),
        # Rod=make_CosseratRod(interpolation="R12", mixed=True, constraints=[0, 1, 2]),
        Rod=make_CosseratRod(
            mixed=True, polynomial_degree=2
        ),
        n_load_steps=10,
        nelements=10,
        VTK_export=True,
        name="Cosserat mixed",
    )
