import numpy as np
from timeit import timeit

from cardillo.rods import (
    CircularCrossSection,
    Simo1986,
    CrossSectionInertias,
    CrossSectionInertias_new,
)
from cardillo.rods._material_models_new import Simo1986 as Simo1986_new
from cardillo import System
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.rods.boostedCosseratRod import make_BoostedCosseratRod
from cardillo.math.rotations import Exp_SO3_quat
from cardillo.math.approx_fprime import approx_fprime
from cardillo.solver import SolverOptions


def get_permutation(nnodes, nnodes_element, n_per_node):
    ################
    # permutations #
    ################
    # from ordering componentwise (old) to nodewise (new)
    i = np.arange(nnodes)[:, None]
    j = np.arange(n_per_node)[None, :]

    idx_new = j + i * n_per_node
    idx_old = i + j * nnodes

    permutation_comp2node = idx_old.ravel()[np.argsort(idx_new.ravel())]
    permutation_node2comp = idx_new.ravel()[np.argsort(idx_old.ravel())]

    # element wise
    i = np.arange(nnodes_element)[:, None]
    j = np.arange(n_per_node)[None, :]

    idx_new = j + i * n_per_node
    idx_old = i + j * nnodes_element

    permutation_comp2node_el = idx_old.ravel()[np.argsort(idx_new.ravel())]
    permutation_node2comp_el = idx_new.ravel()[np.argsort(idx_old.ravel())]

    return dict(
        permutation_comp2node=permutation_comp2node,
        permutation_node2comp=permutation_node2comp,
        permutation_comp2node_el=permutation_comp2node_el,
        permutation_node2comp_el=permutation_node2comp_el,
    )


def test_implementation(n_test=1_000):
    Ei = np.array([1.0, 2.0, 3.0])
    Fi = np.array([40.0, 50.0, 60.0])
    constitutive_law = Simo1986(Ei, Fi)
    constitutive_law_new = Simo1986_new(Ei, Fi)
    cross_section = CircularCrossSection(0.1)
    A_rho0 = np.random.rand()
    B_I_rho0 = np.diag(np.random.rand(3))
    cross_section_inertia = CrossSectionInertias(A_rho0=A_rho0, B_I_rho0=B_I_rho0)
    cross_section_inertia_new = CrossSectionInertias_new(
        A_rho0=A_rho0, B_I_rho0=B_I_rho0
    )
    nelement = 4
    polynomial_degree = 2
    constraints = [0, 1, 5]
    constraints = [0, 1]
    constraints = []

    nquadrature_dyn = int(np.ceil((polynomial_degree + 1) ** 2 / 2))

    # mixed = True
    mixed = False
    if not mixed:
        idx_db = np.setdiff1d(np.arange(6), constraints)
    else:
        idx_db = []
    Rod_old = make_CosseratRod(
        interpolation="Quaternion",
        mixed=mixed,
        polynomial_degree=polynomial_degree,
        constraints=None if len(constraints) == 0 else constraints,
    )
    Rod_new = make_BoostedCosseratRod(
        polynomial_degree=polynomial_degree,
        idx_constraints=constraints,
        idx_displacement_based=idx_db,
        quadrature_dyn=(nquadrature_dyn, "Gauss"),
    )

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
        constitutive_law_new,
        nelement,
        Q=q0_new,
        cross_section_inertias=cross_section_inertia_new,
    )

    # get permutations
    perm_q = get_permutation(rod_old.nnodes_r, rod_old.nnodes_element_r, 7)
    perm_n2c_q = perm_q["permutation_node2comp"]
    perm_c2n_q = perm_q["permutation_comp2node"]
    perm_u = get_permutation(rod_old.nnodes_r, rod_old.nnodes_element_r, 6)
    perm_n2c_u = perm_u["permutation_node2comp"]
    perm_c2n_u = perm_u["permutation_comp2node"]
    if mixed:
        perm_c = get_permutation(
            rod_old.nnodes_la_c, rod_old.nnodes_element_la_c, 6 - len(constraints)
        )
        perm_n2c_c = perm_c["permutation_node2comp"]
        perm_c2n_c = perm_c["permutation_comp2node"]
    else:
        perm_n2c_c = np.arange(0)
        perm_c2n_c = np.arange(0)
        rod_old.la_cDOF = np.arange(0)
    if len(constraints) > 0:
        perm_g = get_permutation(
            rod_old.nnodes_la_g, rod_old.nnodes_element_la_g, len(constraints)
        )
        perm_n2c_g = perm_g["permutation_node2comp"]
        perm_c2n_g = perm_g["permutation_comp2node"]
    else:
        perm_n2c_g = np.arange(0)
        perm_c2n_g = np.arange(0)

    print(f"q0s: {np.linalg.norm(q0_old[perm_c2n_q] - q0_new)}")

    system_old = System()
    system_old.add(rod_old)
    system_old.assemble(
        options=SolverOptions(compute_consistent_initial_conditions=False)
    )

    system_new = System()
    system_new.add(rod_new)
    system_new.assemble(
        options=SolverOptions(compute_consistent_initial_conditions=False)
    )

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
        # constraint equations
        ["g", ("t", "q"), perm_c2n_g, False],
        ["g_q", ("t", "q"), (perm_c2n_g[:, None], perm_c2n_q), True],
        ["W_g", ("t", "q"), (perm_c2n_u[:, None], perm_c2n_g), True],
        ["Wla_g_q", ("t", "q", "la_g"), (perm_c2n_u[:, None], perm_c2n_q), True],
        # dynamic forces
        ["M", ("t", "q"), (perm_c2n_u[:, None], perm_c2n_u), True],
        ["Mu_q", ("t", "q", "u"), (perm_c2n_u[:, None], perm_c2n_q), True],
        ["h", ("t", "q", "u"), perm_c2n_u, False],
        ["h_q", ("t", "q", "u"), (perm_c2n_u[:, None], perm_c2n_q), True],
        ["h_u", ("t", "q", "u"), (perm_c2n_u[:, None], perm_c2n_u), True],
        # quaternion constraint
        ["g_S", ("t", "q"), (), False],
        ["g_S_q", ("t", "q"), (..., perm_c2n_q), True],
        # energies and momentum
        ["E_pot_comp", ("t", "q", "la_c"), (), False],
        ["E_pot", ("t", "q"), (), False],
        ["E_kin", ("t", "q", "u"), (), False],
        ["linear_momentum", ("t", "q", "u"), (), False],
        ["angular_momentum", ("t", "q", "u"), (), False],
    ]
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

    for i in range(n_test):
        t_test = np.random.rand()
        q_test__new = system_new.q0 + np.random.rand(system_new.nq) * 5
        u_test__new = system_new.u0 + np.random.rand(system_new.nu) * 5
        u_dot_test__new = system_new.u0 + np.random.rand(system_new.nu) * 5
        la_c_test__new = system_new.la_c0 + np.random.rand(system_new.nla_c) * 5
        la_g_test__new = system_new.la_g0 + np.random.rand(system_new.nla_g) * 5

        # TODO: maybe step callback!
        arguments__new = dict(
            t=t_test,
            q=q_test__new,
            u=u_test__new,
            u_dot=u_dot_test__new,
            la_c=la_c_test__new,
            la_g=la_g_test__new,
        )
        arguments__old = dict(
            t=t_test,
            q=q_test__new[perm_n2c_q],
            u=u_test__new[perm_n2c_u],
            u_dot=u_dot_test__new[perm_n2c_u],
            la_c=la_c_test__new[perm_n2c_c],
            la_g=la_g_test__new[perm_n2c_g],
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

                if isinstance(result, (float, int)):
                    results[-1] = np.array(result)

            error = np.linalg.norm(results[0] - results[1][permutation])
            if error > 1e-10:
                print(f"{function_name}: {error}")

        # interactions
        for function_name, argument_names in interactions:
            for xi in np.linspace(0.0, 1.0, 2 * rod_new.nnodes):
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
            for xi in np.linspace(0.0, 1.0, rod_new.nnodes * 2):
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


def compare_performance(n_test=1_000):
    Ei = np.array([1.0, 2.0, 3.0])
    Fi = np.array([40.0, 50.0, 60.0])
    constitutive_law = Simo1986(Ei, Fi)
    constitutive_law_new = Simo1986_new(Ei, Fi)
    cross_section = CircularCrossSection(0.1)
    A_rho0 = np.random.rand()
    B_I_rho0 = np.diag(np.random.rand(3))
    cross_section_inertia = CrossSectionInertias(A_rho0=A_rho0, B_I_rho0=B_I_rho0)
    cross_section_inertia_new = CrossSectionInertias_new(
        A_rho0=A_rho0, B_I_rho0=B_I_rho0
    )
    nelement = 500
    polynomial_degree = 2

    mixed = False
    constraints = [0, 1, 5]
    Rod_old = make_CosseratRod(
        interpolation="Quaternion",
        mixed=mixed,
        polynomial_degree=polynomial_degree,
        constraints=constraints,
    )
    Rod_new = make_BoostedCosseratRod(
        polynomial_degree=polynomial_degree,
        idx_constraints=constraints,
        idx_displacement_based=[] if mixed else np.setdiff1d(np.arange(6), constraints),
    )

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
        constitutive_law_new,
        nelement,
        Q=q0_new,
        cross_section_inertias=cross_section_inertia_new,
    )

    system_old = System()
    system_old.add(rod_old)
    system_old.assemble(
        options=SolverOptions(compute_consistent_initial_conditions=False)
    )

    system_new = System()
    system_new.add(rod_new)
    system_new.assemble(
        options=SolverOptions(compute_consistent_initial_conditions=False)
    )

    # get permutations
    perm_q = get_permutation(rod_old.nnodes_r, rod_old.nnodes_element_r, 7)
    perm_n2c_q = perm_q["permutation_node2comp"]
    perm_u = get_permutation(rod_old.nnodes_r, rod_old.nnodes_element_r, 6)
    perm_n2c_u = perm_u["permutation_node2comp"]
    if len(constraints) < 6 and mixed:
        perm_c = get_permutation(
            rod_old.nnodes_la_c, rod_old.nnodes_element_la_c, 6 - len(constraints)
        )
        perm_n2c_c = perm_c["permutation_node2comp"]
    else:
        perm_n2c_c = np.arange(0)
    if len(constraints) > 0:
        perm_g = get_permutation(
            rod_old.nnodes_la_g, rod_old.nnodes_element_la_g, len(constraints)
        )
        perm_n2c_g = perm_g["permutation_node2comp"]
    else:
        perm_n2c_g = np.arange(0)

    # fmt: off
    functions = [
        # kinematic equation
        ["q_dot", ("t", "q", "u"), False],
        ["q_dot_q", ("t", "q", "u"), False],
        # ["q_dot_u", ("t", "q"), False],               # q_dot_u of old is very slow!
        # # compliance equation
        # ["la_c", ("t", "q", "u"), False],
        # ["c", ("t", "q", "u", "la_c"), False],
        # ["c_q", ("t", "q", "u", "la_c"), False],
        # # ["c_u", ("t", "q", "u", "la_c"), False],      # rod has no c_u (only system)
        # # ["c_la_c", (), False],                        # c_la_c is a getter function
        # ["W_c", ("t", "q"), False],
        # ["Wla_c_q", ("t", "q", "la_c"), False],
        # constraint equation
        ["g", ("t", "q"), False],
        ["g_q", ("t", "q"), False],
        ["W_g", ("t", "q"), False],
        ["Wla_g_q", ("t", "q", "la_g"), False],
        # dynamic forces
        # ["M", ("t", "q"), False],                     # M is a getter function
        # ["Mu_q", ("t", "q", "u"), False],             # rod has no Mu_q (only system)
        ["h", ("t", "q", "u"), False],
        ["h_q", ("t", "q", "u"), False],              # rod has no h_q (only system)
        ["h_u", ("t", "q", "u"), False],
        # quaternion constraint
        ["g_S", ("t", "q"), False],
        ["g_S_q", ("t", "q"), False],
        ################
        # interactions #
        ################
        ["r_OP", ("t", "q", "xi", "B_r_CP"), True],
        ["v_P", ("t", "q", "u", "xi", "B_r_CP"), True],
        ["a_P", ("t", "q", "u", "u_dot", "xi", "B_r_CP"), True],

        ["A_IB", ("t", "q", "xi"), True], # this one is cached in cardillo
        ["B_Omega", ("t", "q", "u", "xi"), True],
        ["B_Psi", ("t", "q", "u", "u_dot", "xi"), True],

        ["J_P", ("t", "q", "xi", "B_r_CP"), True],
        ["J_P_q", ("t", "q", "xi", "B_r_CP"), True],
        ["r_OP_q", ("t", "q", "xi", "B_r_CP"), True],
        ["v_P_q", ("t", "q", "u", "xi", "B_r_CP"), True],
        ["a_P_q", ("t", "q", "u", "u_dot", "xi", "B_r_CP"), True],
        ["a_P_u", ("t", "q", "u", "u_dot", "xi", "B_r_CP"), True],

        ["B_J_R", ("t", "q", "xi"), True],
        ["B_J_R_q", ("t", "q", "xi"), True],
        ["A_IB_q", ("t", "q", "xi"), True], # this one is cached in cardillo
        ["B_Omega_q", ("t", "q", "u", "xi"), True],
        ["B_Psi_q", ("t", "q", "u", "u_dot", "xi"), True],
        ["B_Psi_u", ("t", "q", "u", "u_dot", "xi"), True],
    ]
    # functions = [
    #     ["c_q", ("t", "q", "u", "la_c"), False],
    #     ["W_c", ("t", "q"), False],
    #     ["Wla_c_q", ("t", "q", "la_c"), False],
    #     ["h_u", ("t", "q", "u"), False],
    # ]
    # functions = [
    #     # ["q_dot_u", ("t", "q"), False],
    #     ["Wla_c_q", ("t", "q", "la_c"), False],
    # ]
    # fmt: on

    t_test = np.random.rand()
    q_test__new = system_new.q0 + np.random.rand(system_new.nq) * 5
    u_test__new = system_new.u0 + np.random.rand(system_new.nu) * 5
    u_dot_test__new = system_new.u0 + np.random.rand(system_new.nu) * 5
    la_c_test__new = system_new.la_c0 + np.random.rand(system_new.nla_c) * 5
    la_g_test__new = system_new.la_g0 + np.random.rand(system_new.nla_g) * 5
    xi = np.random.rand()
    B_r_CP = np.random.rand(3)

    # xi = 0.0
    # B_r_CP = np.zeros(3)

    # TODO: maybe step callback!
    arguments__new = dict(
        t=t_test,
        q=q_test__new,
        u=u_test__new,
        u_dot=u_dot_test__new,
        la_c=la_c_test__new,
        la_g=la_g_test__new,
        xi=xi,
        B_r_CP=B_r_CP,
    )
    arguments__old = dict(
        t=t_test,
        q=q_test__new[perm_n2c_q],
        u=u_test__new[perm_n2c_u],
        u_dot=u_dot_test__new[perm_n2c_u],
        la_c=la_c_test__new[perm_n2c_c],
        la_g=la_g_test__new[perm_n2c_g],
        xi=xi,
        B_r_CP=B_r_CP,
    )

    for function_name, argument_names, is_interaction in functions:
        function_new = getattr(rod_new, function_name)
        function_old = getattr(rod_old, function_name)

        args_old = []
        args_new = []
        for name in argument_names:
            if "xi" in argument_names:
                if name in ["q", "u", "u_dot"]:
                    if name == "q":
                        DOF_old = rod_old.local_qDOF_P(xi)
                        DOF_new = rod_new.local_qDOF_P(xi)
                    else:
                        DOF_old = rod_old.local_uDOF_P(xi)
                        DOF_new = rod_new.local_uDOF_P(xi)

                    args_old.append(arguments__old[name][DOF_old])
                    args_new.append(arguments__new[name][DOF_new])
                else:
                    args_old.append(arguments__old[name])
                    args_new.append(arguments__new[name])
            else:
                args_old.append(arguments__old[name])
                args_new.append(arguments__new[name])

        def implementation_old():
            return function_old(*args_old)

        def implementation_new():
            return function_new(*args_new)

        # run once
        implementation_old()
        implementation_new()

        N = 1_000 * n_test if is_interaction else n_test

        t_old = timeit(implementation_old, number=N)
        t_new = timeit(implementation_new, number=N)

        ratio = t_new / t_old
        result_str = f"{function_name:<25}: t_new: {t_new:8.3f}, t_old: {t_old:8.3f}, ratio (t_new/t_old): {(t_new / t_old):8.3f}"

        if ratio >= 1.0:
            result_str += f"   --> no improvement!"
        else:
            result_str += f"   --> speedup: {(1 - ratio) * 100:.2f}%!"

        print(result_str)


if __name__ == "__main__":
    print("Test single")
    test_implementation(n_test=1)
    # print("Test 1_000")
    # test_implementation(n_test=1_000)
    print("Compare performance")
    compare_performance(n_test=20)
