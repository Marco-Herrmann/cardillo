import numpy as np

from cardillo.rods import CircularCrossSection, Simo1986, CrossSectionInertias
from cardillo import System
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.rods.boostedCosseratRod import make_BoostedCosseratRod
from cardillo.math.rotations import Exp_SO3_quat

if __name__ == "__main__":
    constitutive_law = Simo1986(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    cross_section = CircularCrossSection(0.1)
    A_rho0 = np.random.rand()
    B_I_rho0 = np.diag(np.random.rand(3))
    cross_section_inertia = CrossSectionInertias(A_rho0=A_rho0, B_I_rho0=B_I_rho0)
    nelement = 2
    polynomial_degree = 2

    Rod_old = make_CosseratRod(
        interpolation="Quaternion", mixed=True, polynomial_degree=polynomial_degree
    )
    Rod_new = make_BoostedCosseratRod(polynomial_degree=polynomial_degree)

    q0_old = Rod_old.straight_configuration(nelement, 5)
    rod_old = Rod_old(
        cross_section,
        constitutive_law,
        nelement,
        Q=q0_old,
        cross_section_inertias=cross_section_inertia,
    )

    q0_new = Rod_new.straight_configuration(nelement, 5)
    rod_new = Rod_new(
        cross_section,
        constitutive_law,
        nelement,
        Q=q0_new,
        cross_section_inertias=cross_section_inertia,
    )

    print(f"q0s: {np.linalg.norm(q0_old - q0_new)}")

    system_old = System()
    system_old.add(rod_old)
    system_old.assemble()

    system_new = System()
    system_new.add(rod_new)
    system_new.assemble()

    # get permutations
    perm_n2c_q = rod_new.permutation_node2comp_q
    perm_c2n_q = rod_new.permutation_comp2node_q
    perm_n2c_u = rod_new.permutation_node2comp_u
    perm_c2n_u = rod_new.permutation_comp2node_u
    perm_n2c_c = rod_new.permutation_node2comp_c
    perm_c2n_c = rod_new.permutation_comp2node_c

    # fmt: off
    functions = [
        # kinematic equation
        ["q_dot", ("t", "q", "u"), perm_c2n_q, False],
        ["q_dot_q", ("t", "q", "u"), (perm_c2n_q[:, None], perm_c2n_q), True],
        ["q_dot_u", ("t", "q"), (perm_c2n_q[:, None], perm_c2n_u), True],
        # compliance equation
        ["la_c", ("t", "q", "u"), perm_c2n_c, False],
        ["c", ("t", "q", "u", "la_c"), perm_c2n_c, False],
        ["c_q", ("t", "q", "u", "la_c"), (perm_c2n_c[:, None], perm_c2n_q), True],
        ["c_u", ("t", "q", "u", "la_c"), (perm_c2n_c[:, None], perm_c2n_u), True],
        ["c_la_c", (), (perm_c2n_c[:, None], perm_c2n_c), True],
        ["W_c", ("t", "q"), (perm_c2n_u[:, None], perm_c2n_c), True],
        ["Wla_c_q", ("t", "q", "la_c"), (perm_c2n_u[:, None], perm_c2n_q), True],
        # dynamic forces
        ["M", ("t", "q"), (perm_c2n_u[:, None], perm_c2n_u), True],
        ["Mu_q", ("t", "q", "u"), (perm_c2n_u[:, None], perm_c2n_q), True],
        ["h", ("t", "q", "u"), perm_c2n_u, False],
        ["h_q", ("t", "q", "u"), (perm_c2n_u[:, None], perm_c2n_q), True],
        ["h_u", ("t", "q", "u"), (perm_c2n_u[:, None], perm_c2n_u), True],
        # quaternion constraint
        ["g_S", ("t", "q"), (), False],
        ["g_S_q", ("t", "q"), (..., perm_c2n_q), True],
    ]
    # fmt: on

    n_test = 1_000
    for i in range(n_test):
        t_test = np.random.rand()
        q_test__new = system_new.q0 + np.random.rand(system_new.nq) * 5
        u_test__new = system_new.u0 + np.random.rand(system_new.nu) * 5
        la_c_test__new = system_new.la_c0 + np.random.rand(system_new.nla_c) * 5

        # TODO: maybe step callback!
        arguments__new = dict(
            t=t_test,
            q=q_test__new,
            u=u_test__new,
            la_c=la_c_test__new,
        )
        arguments__old = dict(
            t=t_test,
            q=q_test__new[perm_n2c_q],
            u=u_test__new[perm_n2c_u],
            la_c=la_c_test__new[perm_n2c_c],
        )

        for function_name, argument_names, permutation, sparse in functions:
            results = []
            for system, arguments_dict in zip(
                [system_new, system_old], [arguments__new, arguments__old]
            ):
                arguments = [arguments_dict[name] for name in argument_names]
                result = getattr(system, function_name)(*arguments)
                if sparse:
                    results.append(result.toarray())
                else:
                    results.append(result)

            error = np.linalg.norm(results[0] - results[1][permutation])
            if error > 1e-10:
                print(f"{function_name}: {error}")

        # # interactions
        # r_OP0 = q_test__new[:3]
        # r_OP0_ = rod_new.r_OP(
        #     t_test, q_test__new[rod_new.qDOF][rod_new.local_qDOF_P(0.0)], 0.0
        # )

        # A_IB0 = Exp_SO3_quat(q_test__new[3:7])
        # A_IB0_ = rod_new.A_IB(
        #     t_test, q_test__new[rod_new.qDOF][rod_new.local_qDOF_P(0.0)], 0.0
        # )

        # print(np.linalg.norm(r_OP0 - r_OP0_))
        # print(np.linalg.norm(A_IB0 - A_IB0_))

        # r_OP1 = q_test__new[-7:-4]
        # r_OP1_ = rod_new.r_OP(
        #     t_test, q_test__new[rod_new.qDOF][rod_new.local_qDOF_P(1.0)], 1.0
        # )

        # A_IB1 = Exp_SO3_quat(q_test__new[-4:])
        # A_IB1_ = rod_new.A_IB(
        #     t_test, q_test__new[rod_new.qDOF][rod_new.local_qDOF_P(1.0)], 1.0
        # )

        # print(np.linalg.norm(r_OP1 - r_OP1_))
        # print(np.linalg.norm(A_IB1 - A_IB1_))
