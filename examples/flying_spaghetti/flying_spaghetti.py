import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation

from cardillo import System
from cardillo.constraints import RigidConnection, Spherical
from cardillo.discrete import RigidBody, Cylinder
from cardillo.forces import Force, Moment, B_Moment
from cardillo.math import e1, e2, e3, ax2skew
from cardillo.math.rotations import A_IB_basic
from cardillo.rods import (
    CircularCrossSection,
    CrossSectionInertias,
    Simo1986,
    animate_beam,
)
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.rods.force_line_distributed import Force_line_distributed
from cardillo.solver import (
    Newton,
    BackwardEuler,
    SolverOptions,
    ScipyDAE,
    DualStormerVerlet,
)


"""https://link.springer.com/article/10.1007/s11044-025-10087-9"""


def flying_spaghetti(
    Rod,
    nelement=10,
    planar=True,
    VTK_export=True,
):

    constitutive_law = Simo1986(np.array([1e4, 1e4, 1e4]), np.array([2e2, 1e2, 1e2]))
    cross_section_inertias = CrossSectionInertias(
        A_rho0=1.0, B_I_rho0=np.diag([10.0, 10.0, 10.0])
    )

    dx = 6.0
    dy = 8.0
    r_OP0 = np.array([dx, 0.0, 0.0])
    alpha = np.pi - np.atan2(dy, dx)
    length = np.sqrt(dx**2 + dy**2)
    A_IB0 = A_IB_basic(alpha).z

    q0, u0 = Rod.straight_initial_configuration(
        nelement,
        length,
        r_OP0=r_OP0,
        A_IB0=A_IB0,
        v_P0=np.zeros(3, dtype=float),
        B_omega_IB0=np.zeros(3, dtype=float),
    )

    rod = Rod(
        CircularCrossSection(1e-2 * length),  # only vis
        constitutive_law,
        nelement,
        Q=q0,
        q0=q0,
        u0=u0,
        cross_section_inertias=cross_section_inertias,
        name="Spaghetti",
    )

    # forcing
    def forcing(t):
        # returns I_F and B_M
        if t <= 2.5:
            if planar:
                return [
                    np.array([8.0, 0.0, 0.0]),
                    np.array([0.0, 0.0, -80.0]),
                ]
            else:
                return [
                    np.array([8.0, 0.0, 0.0]),
                    np.array([-80.0, 0.0, 0.0]),
                ]
        else:
            return [
                np.zeros(3, dtype=float),
                np.zeros(3, dtype=float),
            ]

    force = Force(lambda t: forcing(t)[0], rod, xi=0.0)
    moment = B_Moment(lambda t: forcing(t)[1], rod, xi=0.0)

    # assemble
    system = System()
    system.add(rod, force, moment)

    system.assemble()

    # solving
    t_end = 15
    # t_end = 1
    dt = 0.1
    # solver = DualStormerVerlet(system, t_end, dt)
    # solver = BackwardEuler(system, t_end, dt)
    solver = ScipyDAE(system, t_end, dt)

    sol = solver.solve()

    # VTK export
    dir_name = Path(__file__).parent
    if VTK_export:
        system.export(dir_name, f"vtk_{'planar' if planar else 'spatial'}", sol, fps=50)

    # postprocessing
    nt = len(sol.t)
    t = sol.t
    q = sol.q
    u = sol.u
    la_c = sol.la_c

    energy = np.zeros((3, nt), dtype=float)
    lin_momentum = np.zeros((3, nt), dtype=float)
    ang_momentum = np.zeros((3, nt), dtype=float)
    for i in range(nt):
        force_pot = force.E_pot(t[i], q[i])
        energy[0, i] = system.E_kin(t[i], q[i], u[i])
        energy[1, i] = system.E_pot(t[i], q[i]) - force_pot
        energy[2, i] = system.E_pot_c(t[i], q[i], la_c[i])

        lin_momentum[:, i] = rod.linear_momentum(t[i], q[i], u[i])
        ang_momentum[:, i] = rod.angular_momentum(t[i], q[i], u[i])

    fig, ax = plt.subplots(2, 2, squeeze=False)

    ax[0, 0].plot(t, energy[0], "r", label="E_kin")
    ax[0, 0].plot(t, energy[1], "k", label="E_pot")
    ax[0, 0].plot(t, energy[0] + energy[1], "b", label="E_kin + E_pot")

    ax[0, 1].plot(t, energy[0], "r", label="E_kin")
    ax[0, 1].plot(t, energy[2], "k", label="E_pot_c")
    ax[0, 1].plot(t, energy[0] + energy[2], "b", label="E_kin + E_pot_c")

    ax[1, 0].plot(t, lin_momentum[0], "r", label="lin. momentum ex")
    ax[1, 0].plot(t, lin_momentum[1], "g", label="lin. momentum ey")
    ax[1, 0].plot(t, lin_momentum[2], "b", label="lin. momentum ez")

    ax[1, 1].plot(t, ang_momentum[0], "r", label="ang. momentum ex")
    ax[1, 1].plot(t, ang_momentum[1], "g", label="ang. momentum ey")
    ax[1, 1].plot(t, ang_momentum[2], "b", label="ang. momentum ez")

    [axii.legend() for axi in ax for axii in axi]
    [axii.grid() for axi in ax for axii in axi]

    fig, ax = plt.subplots(2, 2, squeeze=False)

    ax[0, 0].semilogy(
        t[:-1], np.abs(np.diff(energy[0] + energy[1])), "b", label="diff(E_kin + E_pot)"
    )
    ax[0, 1].semilogy(
        t[:-1],
        np.abs(np.diff(energy[0] + energy[2])),
        "b",
        label="diff(E_kin + E_pot_c)",
    )

    ax[1, 0].semilogy(
        t[:-1], np.abs(np.diff(lin_momentum[0])), "r", label="diff(lin. momentum ex)"
    )
    ax[1, 0].semilogy(
        t[:-1], np.abs(np.diff(lin_momentum[1])), "g", label="diff(lin. momentum ey)"
    )
    ax[1, 0].semilogy(
        t[:-1], np.abs(np.diff(lin_momentum[2])), "b", label="diff(lin. momentum ez)"
    )

    ax[1, 1].semilogy(
        t[:-1], np.abs(np.diff(ang_momentum[0])), "r", label="diff(ang. momentum ex)"
    )
    ax[1, 1].semilogy(
        t[:-1], np.abs(np.diff(ang_momentum[1])), "g", label="diff(ang. momentum ey)"
    )
    ax[1, 1].semilogy(
        t[:-1], np.abs(np.diff(ang_momentum[2])), "b", label="diff(ang. momentum ez)"
    )

    [axii.legend() for axi in ax for axii in axi]
    [axii.grid() for axi in ax for axii in axi]

    plt.show()


if __name__ == "__main__":
    Rod = make_CosseratRod()

    flying_spaghetti(Rod, 10, False, False)
