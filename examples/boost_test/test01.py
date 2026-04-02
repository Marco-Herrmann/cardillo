import numpy as np

from cardillo.rods import CircularCrossSection, Simo1986, CrossSectionInertias
from cardillo import System
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.rods.boostedCosseratRod import make_BoostedCosseratRod
from cardillo.math.rotations import Exp_SO3_quat
from cardillo.math.approx_fprime import approx_fprime

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
    # functions = []
    interactions = [
        ["r_OP", ("t", "q", "B_r_CP")],
        ["v_P", ("t", "q", "u", "B_r_CP")],
        ["a_P", ("t", "q", "u", "u_dot", "B_r_CP")],

        ["A_IB", ("t", "q")],
        ["B_Omega", ("t", "q", "u")],
        ["B_Psi", ("t", "q", "u", "u_dot")],
    ]
    derivatives = [
        [("J_P", ("t", "q", "B_r_CP")), ("v_P", ("t", "q", "u", "B_r_CP")), 2],
        [("J_P_q", ("t", "q", "B_r_CP")), ("J_P", ("t", "q", "B_r_CP")), 1],
        [("r_OP_q", ("t", "q", "B_r_CP")), ("r_OP", ("t", "q", "B_r_CP")), 1],
        [("v_P_q", ("t", "q", "u", "B_r_CP")), ("v_P", ("t", "q", "u", "B_r_CP")), 1],
        [("a_P_q", ("t", "q", "u", "u_dot", "B_r_CP")), ("a_P", ("t", "q", "u", "u_dot", "B_r_CP")), 1],
        [("a_P_u", ("t", "q", "u", "u_dot", "B_r_CP")), ("a_P", ("t", "q", "u", "u_dot", "B_r_CP")), 2],

        [("B_J_R", ("t", "q")), ("B_Omega", ("t", "q", "u")), 2],
        [("B_J_R_q", ("t", "q")), ("B_J_R", ("t", "q")), 1],
        [("A_IB_q", ("t", "q")), ("A_IB", ("t", "q")), 1],
        [("B_Omega_q", ("t", "q", "u")), ("B_Omega", ("t", "q", "u")), 1],
        [("B_Psi_q", ("t", "q", "u", "u_dot")), ("B_Psi", ("t", "q", "u", "u_dot")), 1],
        [("B_Psi_u", ("t", "q", "u", "u_dot")), ("B_Psi", ("t", "q", "u", "u_dot")), 2],
    ]
    # fmt: on

    n_test = 1_000
    for i in range(n_test):
        t_test = np.random.rand()
        q_test__new = system_new.q0 + np.random.rand(system_new.nq) * 5
        u_test__new = system_new.u0 + np.random.rand(system_new.nu) * 5
        u_dot_test__new = system_new.u0 + np.random.rand(system_new.nu) * 5
        la_c_test__new = system_new.la_c0 + np.random.rand(system_new.nla_c) * 5

        # TODO: maybe step callback!
        arguments__new = dict(
            t=t_test,
            q=q_test__new,
            u=u_test__new,
            u_dot=u_dot_test__new,
            la_c=la_c_test__new,
        )
        arguments__old = dict(
            t=t_test,
            q=q_test__new[perm_n2c_q],
            u=u_test__new[perm_n2c_u],
            u_dot=u_dot_test__new[perm_n2c_u],
            la_c=la_c_test__new[perm_n2c_c],
        )

        # system functions
        for function_name, argument_names, permutation, sparse in functions:
            results = []
            for system, arguments_dict in zip(
                [system_new, system_old], [arguments__new, arguments__old]
            ):
                args_i = [arguments_dict[name] for name in argument_names]
                result = getattr(system, function_name)(*args_i)
                if sparse:
                    results.append(result.toarray())
                else:
                    results.append(result)

            error = np.linalg.norm(results[0] - results[1][permutation])
            if error > 1e-10:
                print(f"{function_name}: {error}")

        # interactions
        for function_name, argument_names in interactions:
            for xi in np.linspace(0.0, 1.0, rod_new.nnodes):
                for B_r_CP in [np.zeros(3), np.random.rand(3)]:
                    arguments__new["B_r_CP"] = B_r_CP
                    arguments__old["B_r_CP"] = B_r_CP

                    results = []
                    for rod, arguments_dict in zip(
                        [rod_new, rod_old], [arguments__new, arguments__old]
                    ):
                        kwargs = {name: arguments_dict[name] for name in argument_names}
                        args_i = []
                        if "t" in argument_names:
                            args_i.append(t_test)
                        if "q" in argument_names:
                            qDOF = rod.local_qDOF_P(xi)
                            args_i.append(kwargs["q"][qDOF])

                        if "u" in argument_names:
                            uDOF = rod.local_uDOF_P(xi)
                            args_i.append(kwargs["u"][uDOF])

                        if "u_dot" in argument_names:
                            uDOF = rod.local_uDOF_P(xi)
                            args_i.append(kwargs["u_dot"][uDOF])

                        args_i.append(xi)
                        if "B_r_CP" in argument_names:
                            args_i.append(B_r_CP)

                        results.append(getattr(rod, function_name)(*args_i))

                    error = np.linalg.norm(results[0] - results[1])
                    if error > 1e-10:
                        print(f"{function_name}: {error}")

        # derivatives
        for deriv, orig, i in derivatives:
            for xi in np.linspace(0.0, 1.0, rod_new.nnodes):
                for B_r_CP in [np.zeros(3), np.random.rand(3)]:
                    arguments__new["B_r_CP"] = B_r_CP
                    args = []
                    for a_names in [deriv[1], orig[1]]:
                        kwargs = {name: arguments__new[name] for name in a_names}
                        args_i = []
                        if "t" in a_names:
                            args_i.append(t_test)
                        if "q" in a_names:
                            qDOF = rod_new.local_qDOF_P(xi)
                            args_i.append(kwargs["q"][qDOF])

                        if "u" in a_names:
                            uDOF = rod_new.local_uDOF_P(xi)
                            args_i.append(kwargs["u"][uDOF])

                        if "u_dot" in a_names:
                            uDOF = rod_new.local_uDOF_P(xi)
                            args_i.append(kwargs["u_dot"][uDOF])

                        args_i.append(xi)
                        if "B_r_CP" in a_names:
                            args_i.append(B_r_CP)

                        args.append(args_i)

                    #######################
                    # evaluate derivative #
                    #######################
                    derivative = getattr(rod_new, deriv[0])(*args[0])

                    ##################################
                    # approximate with approx_fprime #
                    ##################################
                    orig_fun = getattr(rod_new, orig[0])
                    approx = approx_fprime(
                        args[1][i],
                        lambda x: orig_fun(*args[1][:i], x, *args[1][i + 1 :]),
                    )

                    error = np.linalg.norm(derivative - approx)
                    if error > 1e-5:
                        print(f"{deriv[0]}: {error}")
