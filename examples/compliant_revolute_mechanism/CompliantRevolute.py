from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.constraints import RigidConnection
from cardillo.discrete import RigidBody, Frame, Meshed
from cardillo.forces import Moment
from cardillo.math import A_IB_basic
from cardillo.rods import RectangularCrossSection, Simo1986, CrossSectionInertias
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.solver import Newton
from cardillo.utility.sensor import Sensor, SensorRecords

if __name__ == "__main__":
    # read directory name of this file
    dir_name = Path(__file__).parent

    width = 1.5
    width = 2.0
    width_string = f"{width:.2f}".replace(".", "p")

    #################################
    # base position and orientation #
    #################################

    # # crossing at 0, oriented as in stl
    # r_OB = 42.0 * np.sqrt(2)/2 * np.array([1.0, 1.0, 0.0]) - width/2 * np.array([1.0, 1.0, 0.0])
    # A_IB = A_IB_basic(-np.pi/2).z @ A_IB_basic(-np.pi).x

    # crossing at 0, oriented with arm upwards
    r_OB = -42.0 * np.sqrt(2) / 2 * np.array([1.0, 1.0, 0.0]) + width / 2 * np.array(
        [1.0, 1.0, 0.0]
    )
    A_IB = np.eye(3, dtype=float)

    # compute all positions for generalized coordinates
    # Q1/Q2: points on the middle of the surface
    # R1/R2: points where the rods are starting
    BT_r_BTQ1 = 42.0 * np.sqrt(2) / 2 * np.array([-1.0, 1.0, 0.0])
    BT_r_Q1R1 = np.array([4.0, -width / 2, 0.0])
    BT_r_BTQ2 = -BT_r_BTQ1
    BT_r_Q2R2 = np.array([-width / 2, 4.0, 0.0])

    BT_r_BTC = 42.0 * np.sqrt(2) / 2 * np.array([1.0, 1.0, 0.0]) - width / 2 * np.array(
        [1.0, 1.0, 0.0]
    )

    # orientations
    A_BT = A_IB_basic(np.pi).x @ A_IB_basic(np.pi / 2).z
    A_IT = A_IB @ A_BT

    # B: basis
    # C: center
    # T: top
    r_OC = r_OB + A_IB @ BT_r_BTC
    r_OT = r_OC - A_IT @ BT_r_BTC

    q0_B = RigidBody.pose2q(r_OB, A_IB)
    q0_T = RigidBody.pose2q(r_OT, A_IT)

    # start and end points of the rod
    dh = np.array([0.0, 0.0, 32.25])
    r_OR1s = r_OB + A_IB @ (BT_r_BTQ1 + BT_r_Q1R1 + dh)
    r_OR1e = r_OT + A_IT @ (BT_r_BTQ2 + BT_r_Q2R2 - dh)

    r_OR2s = r_OB + A_IB @ (BT_r_BTQ1 + BT_r_Q1R1 - dh)
    r_OR2e = r_OT + A_IT @ (BT_r_BTQ2 + BT_r_Q2R2 + dh)

    r_OR3s = r_OB + A_IB @ (BT_r_BTQ2 + BT_r_Q2R2)
    r_OR3e = r_OT + A_IT @ (BT_r_BTQ1 + BT_r_Q1R1)

    l1 = np.linalg.norm(r_OR1s - r_OR1e)
    l2 = np.linalg.norm(r_OR2s - r_OR2e)
    l3 = np.linalg.norm(r_OR3s - r_OR3e)

    # initialize system
    system = System(origin_size=20.0)

    ###########
    # base part
    ###########
    # create base part and add it to system
    base_part = Meshed(Frame)(
        mesh_obj=Path(dir_name, "stl", "base.stl"),
        r_OP=r_OB,
        A_IB=A_IB,
        name="base_part",
    )
    system.add(base_part)

    ##########
    # top part
    ##########
    rho = 1.3 * 1e-6
    top_part = Meshed(RigidBody)(
        mesh_obj=Path(dir_name, "stl", "top.stl"),
        density=rho,
        q0=q0_T,
        name="top_part",
    )

    # add contributions to the system
    system.add(top_part)

    marker_hole = Sensor(
        top_part, B_r_PQ=np.array([-100.0, 0.0, 0.0]), name=f"Hole_{width_string}"
    )
    marker_frame = Sensor(top_part, name=f"Frame_{width_string}")
    system.add(marker_hole, marker_frame)

    # create rods
    Rod = make_CosseratRod()  # constraints=[0, 1, 2])
    nelements = 10

    # cross section, start positions and orientations
    cross_sections = [
        RectangularCrossSection(width, 19.5),
        RectangularCrossSection(width, 19.5),
        RectangularCrossSection(width, 39),
    ]
    r_OP0s = [r_OR1s, r_OR2s, r_OR3s]
    A_IB0s = [A_IB, A_IB, A_IB @ A_IB_basic(np.pi / 2).z]

    for i, (cross_section, r_OP0, A_IB0) in enumerate(
        zip(cross_sections, r_OP0s, A_IB0s, strict=True)
    ):
        A = cross_section.area
        I1, I2, I3 = np.diag(cross_section.second_moment)

        # material model
        E = 2.5 * 1e3
        G = E / 2
        Ei = np.array([E * A, G * A, G * A])
        Fi = np.array([G * I1, E * I2, E * I3])
        material_model = Simo1986(Ei, Fi)

        q0_rod = Rod.straight_configuration(nelements, l1, r_OP0=r_OP0, A_IB0=A_IB0)
        rod = Rod(cross_section, material_model, nelements, Q=q0_rod, name=f"Rod{i+1}")
        system.add(rod)

        # connect rod to parts
        connect_base = RigidConnection(
            base_part, rod, xi2=0.0, name=f"RigidConnection_base_rod{i+1}"
        )
        connect_top = RigidConnection(
            top_part, rod, xi2=1.0, name=f"RigidConnection_top_rod{i+1}"
        )
        system.add(connect_base, connect_top)

    # add external moment at the top part
    k_b = E * I3
    m_max = k_b / 20.0
    print(f"{m_max = }")

    def M_ext(t):
        if t < 1 / 3:
            t_factor = 3 * t
        else:
            t_factor = 1.0 - 3 * (t - 1 / 3)

        return np.array([0.0, 0.0, m_max]) * t_factor

    moment = Moment(M_ext, top_part)
    system.add(moment)

    # assemble system
    system.assemble()

    ############
    # simulation
    ############
    n_load_steps = 99
    solver = Newton(system, n_load_steps)  # create solver
    sol = solver.solve()  # simulate system

    # read solution
    t = sol.t
    q = sol.q

    marker_hole.save(
        dir_name,
        "csv",
        sol,
        [SensorRecords.r_OP],
        save=True,
        plot=False,
    )
    marker_frame.save(
        dir_name,
        "csv",
        sol,
        [SensorRecords.r_OP],
        save=True,
        plot=False,
    )

    #################
    # post-processing
    #################
    r_OP_hole = np.array(
        [
            top_part.r_OP(ti, qi[top_part.qDOF], B_r_CP=np.array([-100.0, 0.0, 0.0]))
            for ti, qi in zip(t, q)
        ]
    )
    r_OP_frame = np.array(
        [top_part.r_OP(ti, qi[top_part.qDOF]) for ti, qi in zip(t, q)]
    )

    radius_hole = np.sqrt(r_OP_hole[0, 0] ** 2 + r_OP_hole[0, 1] ** 2)
    r_OCircle_hole = np.array(
        [
            [radius_hole * np.cos(alpha), radius_hole * np.sin(alpha), 0.0]
            for alpha in np.linspace(0, 2 * np.pi, 100)
        ]
    )

    radius_frame = np.sqrt(r_OP_frame[0, 0] ** 2 + r_OP_frame[0, 1] ** 2)
    r_OCircle_frame = np.array(
        [
            [radius_frame * np.cos(alpha), radius_frame * np.sin(alpha), 0.0]
            for alpha in np.linspace(0, 2 * np.pi, 100)
        ]
    )

    fig, ax = plt.subplots()
    ax.plot(r_OP_hole[:, 0], r_OP_hole[:, 1], label="Hole position")
    ax.plot(r_OP_frame[:, 0], r_OP_frame[:, 1], label="Frame position")
    ax.plot(r_OCircle_hole[:, 0], r_OCircle_hole[:, 1], label="Reference circle hole")
    ax.plot(
        r_OCircle_frame[:, 0], r_OCircle_frame[:, 1], label="Reference circle frame"
    )
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.legend()
    ax.grid()
    ax.set_title(width_string)
    ax.set_aspect("equal", adjustable="box")

    plt.show()

    # vtk-export
    dir_name = Path(__file__).parent
    system.export(dir_name, "vtk", sol)
