import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import warnings

from cardillo import System
from cardillo.constraints import RigidConnection
from cardillo.forces import Force, B_Force, Moment, B_Moment
from cardillo.math import e1, e2, e3, A_IB_basic

from cardillo.rods_new import (
    Simo1986,
    RectangularCrossSection
)
from cardillo.rods_new2 import make_CosseratRod
from cardillo.solver import Newton, SolverOptions
from cardillo.utility.sensor import Sensor, SensorRecords


def fork_structure(
    Rod,
    *,
    nelements_per_segment: int = 3,
    reversal: bool = False,
    #
    n_load_steps_per_configuration: int = 5,
):
    """Experiment 4.7 from https://www.springerprofessional.de/content/pdfId/52920026/10.1007/s00707-026-04768-5"""

    # geometry
    L = 1.0

    cross_section = RectangularCrossSection(L / 10, L / 10)

    # centerline of the curved: 
    #   semi circle with rod radius L, 
    #   starting at (2L, L, 0) 
    #   ending at (2L, -L, 0) 
    #   via (L, 0, 0)
    r_OP02 = lambda xi: L * np.array([2 - np.sin(xi), np.cos(xi), 0.0])
    r_OP02_xi = lambda xi: L * np.array([-np.cos(xi), -np.sin(xi), 0.0])
    r_OP02_xixi = lambda xi: L * np.array([np.sin(xi), -np.cos(xi), 0.0])

    q0s = [
        Rod.straight_configuration(nelements_per_segment, L), 
        Rod.serret_frenet_configuration(2 * nelements_per_segment, r_OP02, r_OP02_xi, r_OP02_xixi, xi1=np.pi),
    ]

    # material model
    Ei = np.array([1e4, 1e4, 1e4])
    Fi = np.array([1e2, 1e2, 1e2])
    material_model = Simo1986(Ei, Fi)

    # create system
    system = System()

    rods = [
        Rod(cross_section, material_model, nelements_per_segment * (i + 1), Q=q0s[i], name=f"rod_{i}") for i in range(len(q0s))
    ]
    system.add(*rods)


    # forces
    Fmax = 200
    F = lambda t: 2 * t * Fmax

    if not reversal: 
        F1 = lambda t: e3 * (F(t) if t <= 0.5 else Fmax)
        F2 = lambda t: -e3 * (F(t - 0.5) if t >= 0.5 else 0.0)     
    else: 
        F1 = lambda t: e3 * (F(t - 0.5) if t >= 0.5 else 0.0)
        F2 = lambda t: -e3 * (F(t) if t <= 0.5 else Fmax)
        
    Forces = [
        Force(F1, rods[1], xi=0, name="Force_1"),
        Force(F2, rods[1], xi=1, name="Force_2")
    ]
    system.add(*Forces)
    
    # constraints 
    constraints = [
        RigidConnection(rods[0], system.origin, xi1=0, name="rod_origin"),
        RigidConnection(rods[0], rods[1], xi1=1, xi2=0.5, name="rod_rod"),
    ]
    system.add(*constraints)

    # sensors
    sensors = [
        Sensor(rods[1], xi=0.0, name="P_1"),
        Sensor(rods[1], xi=1.0, name="P_2"),
        Sensor(rods[1], xi=0.5, name="P_B"),
    ]
    system.add(*sensors)

    # assemble system
    system.assemble()

    # solve
    solver = Newton(
        system,
        n_load_steps=2 * n_load_steps_per_configuration,
        options=SolverOptions(newton_max_iter=50),
    )
    sol = solver.solve()

    #################
    # post-processing
    #################
    # vtk-export
    dir_name = Path(sys.argv[0]).parent
    system.export_blender(dir_name, f"blender", sol, create_blend=True)

    # csv export
    [sensor.save(dir_name, f"csv", sol, functions=[SensorRecords.r_OP]) for sensor in sensors]
    
    # load csv
    poss = ["P_1", "P_2", "P_B"]
    r_OPs = [
        np.loadtxt(dir_name / f"csv/{pos}.csv", delimiter=",", skiprows=1) for pos in poss
    ]

    # Table 4
    u1_P1 = r_OPs[0][n_load_steps_per_configuration, 1:] - r_OPs[0][0, 1:]
    u2_P1 = r_OPs[0][2*n_load_steps_per_configuration, 1:] - r_OPs[0][0, 1:]
    print(f"Table 4: p={rods[0]._polynomial_degree}, nel (tot): {3 * nelements_per_segment}")
    print(u1_P1, u2_P1)

    styles = ["--b", "--r", "--g"] if reversal else ["-b", "-r", "-g"]
    fig, ax = plt.subplots(1, 3, squeeze=False)
    for i in range(3): # [x, y, z]
        for j in range(3): # [P_1, P_2, P_B]
            u = r_OPs[j][:, i+1] - r_OPs[j][0, i+1]
            ax[0, i].plot(r_OPs[j][:, 0], u, styles[j], label=poss[j])

        ax[0, i].grid()
        ax[0, i].legend()
        ax[0, i].set_ylim(-2.0, 2.0)

    plt.show()

if __name__ == "__main__":
    Rod = make_CosseratRod(
        # idx_displacement_based=[0, 1, 2],
        # idx_displacement_based=[3, 4, 5],
        # idx_displacement_based=[0, 1, 2, 3, 4, 5],
    )
    fork_structure(Rod, reversal=False)