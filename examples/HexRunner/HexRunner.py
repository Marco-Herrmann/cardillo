import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cardillo import System
from cardillo.actuators import Motor, PDcontroller
from cardillo.discrete import Frame, RigidBody, Box, Cylinder, PointMass
from cardillo.forces import Force
from cardillo.constraints import Revolute, Prismatic, RigidConnection
from cardillo.math import A_IB_basic, cross3
from cardillo.contacts import Sphere2Plane
from cardillo.interactions import TwoPointInteraction
from cardillo.force_laws import KelvinVoigtElement

from cardillo.solver import Moreau, Rattle, SolverOptions
from cardillo.visualization import Renderer

if __name__ == "__main__":
    # Create the system
    system = System()
    objects = []

    # ground
    floor = Box(Frame)(
        dimensions=[100.0, 2.0, 0.0001],
        name="floor",
        A_IB=np.eye(3),
    )
    system.add(floor)

    # length of the leg
    l_leg = 0.5
    height_offset = 0.01
    v0 = 10.0

    r_Center = np.array([-10.0, 0.0, l_leg + height_offset])
    v_Center = np.array([v0, 0.0, 0.0])
    Omega0 = np.array([0.0, v0 / l_leg, 0.0])

    # lower hanging cylinder
    r_C1 = r_Center - np.array([0.0, 0.0, 0.35])
    A_IB1 = A_IB_basic(np.pi / 2).y
    radius_C1 = 0.05
    length_C1 = 0.5
    m_C1 = 2.0
    B1_Theta_C1 = m_C1 * np.diag(
        [
            1 / 2 * radius_C1**2,
            1 / 12 * (3 * radius_C1**2 + length_C1**2),
            1 / 12 * (3 * radius_C1**2 + length_C1**2),
        ]
    )
    q0_C1 = RigidBody.pose2q(r_C1, A_IB1)
    u0_C1 = np.zeros(6)
    u0_C1[:3] = v_Center
    cylinder1 = Cylinder(RigidBody)(
        radius_C1,
        length_C1,
        None,
        mass=m_C1,
        B_Theta_C=B1_Theta_C1,
        q0=q0_C1,
        u0=u0_C1,
        name="Cylinder1",
    )
    objects.append(cylinder1)

    # left rotating disc
    r_C2 = r_Center + np.array([0.0, 0.1, 0.0])
    A_IB2 = A_IB_basic(np.pi / 2).x
    radius_C2 = 0.05
    length_C2 = 0.02
    m_C2 = 0.5
    B2_Theta_C2 = m_C2 * np.diag(
        [
            1 / 2 * radius_C2**2,
            1 / 12 * (3 * radius_C2**2 + length_C2**2),
            1 / 12 * (3 * radius_C2**2 + length_C2**2),
        ]
    )
    q0_C2 = RigidBody.pose2q(r_C2, A_IB2)
    u0_C2 = np.zeros(6)
    u0_C2[:3] = v_Center
    u0_C2[3:] = A_IB2.T @ Omega0
    cylinder2 = Cylinder(RigidBody)(
        radius_C2,
        length_C2,
        None,
        mass=m_C2,
        B_Theta_C=B2_Theta_C2,
        q0=q0_C2,
        u0=u0_C2,
        name="Cylinder2",
    )
    objects.append(cylinder2)

    # right rotating disc
    r_C3 = r_Center - np.array([0.0, 0.1, 0.0])
    A_IB3 = A_IB_basic(np.pi / 2).x
    q0_C3 = RigidBody.pose2q(r_C3, A_IB3)
    u0_C3 = np.zeros(6)
    u0_C3[:3] = v_Center
    u0_C3[3:] = A_IB3.T @ Omega0
    cylinder3 = Cylinder(RigidBody)(
        radius_C2,
        length_C2,
        None,
        mass=m_C2,
        B_Theta_C=B2_Theta_C2,
        q0=q0_C3,
        u0=u0_C3,
        name="Cylinder3",
    )
    objects.append(cylinder3)

    # connect cylinder1 and cylinder2
    revolute_12 = Revolute(
        cylinder1, cylinder2, axis=1, r_OJ0=r_Center, name="revolute_12"
    )
    system.add(revolute_12)

    rc = RigidConnection(cylinder2, cylinder3, name="rigid_connection")
    system.add(rc)

    # Motor
    print(0.2 * cylinder1.mass * 9.81)
    motor = Motor(revolute_12, lambda t: 4.0)
    system.add(motor)

    contacts = []
    mu = 0.3
    ###############################
    # contacts with rotating axis #
    ###############################
    # # add ground contacts
    # n_leg = 6
    # for i in range(n_leg):
    #     alpha = (30 + 360/n_leg*i) * np.pi/180
    #     B_r_CP = l_leg * np.array([np.cos(alpha), np.sin(alpha), 0.0])
    #     contact = Sphere2Plane(system.origin, cylinder2, mu=mu, e_N=0.0, r=0.0, B_r_CP=B_r_CP, name=f"contact_right_{i}")
    #     contacts.append(contact)

    #     alpha = (-0 + 360/n_leg*i) * np.pi/180
    #     B_r_CP = l_leg * np.array([np.cos(alpha), np.sin(alpha), 0.0])
    #     contact = Sphere2Plane(system.origin, cylinder3, mu=mu, e_N=0.0, r=0.0, B_r_CP=B_r_CP, name=f"contact_left_{i}")
    #     contacts.append(contact)

    ######################
    # contacts with legs #
    ######################
    radius_leg = 0.01
    length_leg = l_leg / 2
    m_leg = 0.05
    Bi_Theta_Ci = m_leg * np.diag(
        [
            1 / 2 * radius_leg**2,
            1 / 12 * (3 * radius_leg**2 + length_leg**2),
            1 / 12 * (3 * radius_leg**2 + length_leg**2),
        ]
    )
    for i, (alpha0, parent) in enumerate(zip([0, 60], [cylinder2, cylinder3])):
        for j in range(3):
            alpha = (alpha0 + 120 * j) * np.pi / 180
            A_IBi = A_IB_basic(alpha).y
            r_Ci = (
                r_Center
                + (-1) ** i * np.array([0.0, 0.1, 0.0])
                + A_IBi @ np.array([0.0, 0.0, 3 / 4 * l_leg])
            )

            q0_Ci = RigidBody.pose2q(r_Ci, A_IBi)
            u0_Ci = np.zeros(6)
            u0_Ci[:3] = v_Center + cross3(Omega0, r_Ci - r_Center)
            u0_Ci[3:] = A_IBi.T @ Omega0
            cylinderi = Cylinder(RigidBody)(
                radius_leg,
                length_leg,
                None,
                mass=m_leg,
                B_Theta_C=Bi_Theta_Ci,
                q0=q0_Ci,
                u0=u0_Ci,
                name=f"Leg_{i}_{j}",
            )
            objects.append(cylinderi)

            # connect cylinderi and parent
            prismatic = Prismatic(
                cylinderi, parent, axis=2, r_OJ0=r_Ci
            )  # , name=f"prismatic_{i}")
            system.add(prismatic)

            # spring-damper
            spring_damper = KelvinVoigtElement(
                TwoPointInteraction(cylinderi, parent),
                k=500.0,
                d=1.0,
                name=f"spring_damper_{i}_{j}",
            )
            system.add(spring_damper)

            # contacts
            contact_ij = Sphere2Plane(
                system.origin,
                cylinderi,
                mu=mu,
                e_N=0.0,
                r=0.0,
                B_r_CP=np.array([0.0, 0.0, l_leg / 4]),
                name=f"contact_{i}_{j}",
            )
            contacts.append(contact_ij)

    # add gravities
    gravity = np.array([0, 0, -9.81])
    for obj in objects:
        system.add(Force(gravity * obj.mass, obj, name=f"{obj.name}_gravity"))

    # assemble system
    system.add(*objects)
    system.add(*contacts)
    system.assemble()

    ############
    # simulation
    ############
    t1 = 2.0
    dt = 1e-3  # time step
    solver = Moreau(
        system,
        t1,
        dt,
        options=SolverOptions(
            continue_with_unconverged=True, fixed_point_max_iter=int(1e4)
        ),
    )
    # solver = Rattle(system, t1, dt)

    render = Renderer(system, [*objects, *contacts, floor])
    # render.start_step_render()

    sol = solver.solve()

    # render.stop_step_render()
    render.render_solution(sol, repeat=True)

    # get energeies of the system
    E_pot = [system.E_pot(ti, qi) for ti, qi in zip(sol.t, sol.q)]
    E_kin = [system.E_kin(ti, qi, ui) for ti, qi, ui in zip(sol.t, sol.q, sol.u)]
    E_tot = [E_kin[i] + E_pot[i] for i in range(len(sol.t))]

    fig, ax = plt.subplots()
    ax.plot(sol.t, E_pot, label="E_pot")
    ax.plot(sol.t, E_kin, label="E_kin")
    ax.plot(sol.t, E_tot, label="E_tot")
    ax.legend()
    ax.grid()

    plt.show()

    # vtk-export
    dir_name = Path(__file__).parent
    system.export(dir_name, "vtk", sol)
