from abc import ABC, abstractmethod
import numpy as np
from vtk import VTK_LAGRANGE_CURVE
from warnings import warn

from cardillo.utility.bezier import L2_projection_Bezier_curve
from cardillo.math import cross3, norm

from ._cross_section import (
    ExportableCrossSection,
    CircularCrossSection,
    RectangularCrossSection,
)

"""
very usefull for the ordering: 
https://coreform.com/papers/implementation-of-rational-bezier-cells-into-VTK-report.pdf
"""


class RodExportBase(ABC):
    def __init__(self, cross_section: ExportableCrossSection):
        self.cross_section = cross_section

        self._export_dict = {
            # "level": "centerline + directors",
            "level": "volume",
            "num_per_cell": "Auto",
            "ncells": "Auto",
            "stresses": False,
            "volume_directors": False,
            "surface_normals": False,
            "hasCap": False,
        }

        self.preprocessed_export = False

    @abstractmethod
    def r_OP(self, t, q, xi, B_r_CP): ...

    @abstractmethod
    def A_IB(self, t, q, xi): ...

    def centerline(self, q, num=100):
        q_body = q[self.qDOF]
        r = []
        for xi in np.linspace(0, 1, num):
            qp = q_body[self.local_qDOF_P(xi)]
            r.append(self.r_OP(1, qp, xi))
        return np.array(r).T

    def frames(self, q, num=10):
        q_body = q[self.qDOF]
        r = []
        d1 = []
        d2 = []
        d3 = []

        for xi in np.linspace(0, 1, num=num):
            qp = q_body[self.local_qDOF_P(xi)]
            r.append(self.r_OP(1, qp, xi))

            d1i, d2i, d3i = self.A_IB(1, qp, xi).T
            d1.extend([d1i])
            d2.extend([d2i])
            d3.extend([d3i])

        return np.array(r).T, np.array(d1).T, np.array(d2).T, np.array(d3).T

    def surface(self, q, xi, eta, el):
        q_body = q[self.qDOF]
        qp = q_body[self.elDOF[el]]
        N, N_xi = self.basis_functions_r(xi, el)
        eval = self._eval(qp, xi, N, N_xi)
        r_OP = eval[0]
        A_IB = eval[1]
        # B_gamma = eval[2]
        # B_kappa_IB = eval[3]

        B_r_PQ = self.cross_section.B_r_PQ(eta)

        r_OQ = r_OP + A_IB @ B_r_PQ
        return r_OQ

    def surface_normal(self, q, xi, eta, el):
        q_body = q[self.qDOF]
        qp = q_body[self.elDOF[el]]
        N, N_xi = self.basis_functions_r(xi, el)
        eval = self._eval(qp, xi, N, N_xi)
        # r_OP = eval[0]
        A_IB = eval[1]
        B_gamma = eval[2]
        B_kappa_IB = eval[3]

        B_r_PQ = self.cross_section.B_r_PQ(eta)
        B_r_PQ_eta = self.cross_section.B_r_PQ_eta(eta)

        # r_OQ = r_OP + A_IB @ B_r_PQ
        r_OQ_xi = A_IB @ B_gamma + A_IB @ cross3(B_kappa_IB, B_r_PQ)
        r_OQ_eta = A_IB @ B_r_PQ_eta

        n = cross3(r_OQ_eta, r_OQ_xi)
        return n / norm(n)

    def preprocess_export(self):
        ########################
        # evaluate export dict #
        ########################
        # number if cells to be exported
        if self._export_dict["ncells"] == "Auto":
            self._export_dict["ncells"] = self.nelement

        assert isinstance(self._export_dict["ncells"], int)
        assert self._export_dict["ncells"] >= 1
        ncells = self._export_dict["ncells"]

        # continuity for L2 projection
        self._export_dict["continuity"] = "C0"

        # what shall be exported
        assert self._export_dict["level"] in [
            "centerline + directors",
            "volume",
            "None",
            None,
        ]

        ########################################################
        # get number of frames for export and make cell-arrays #
        ########################################################
        if self._export_dict["level"] == "centerline + directors":
            # TODO: maybe export directors with respect to their polynomialdegree (to see them at their nodes)
            # for polynomial shape functions this should be the best.
            # But for other interpolations (SE3) this is not good!

            num_per_cell = self._export_dict["num_per_cell"]
            if num_per_cell == "Auto":
                p = self.polynomial_degree_r
            else:
                p = num_per_cell - 1

            # compute number of frames
            self._export_dict["num_frames"] = p * ncells + 1

            # create cells
            self._export_dict["cells"] = [
                (
                    VTK_LAGRANGE_CURVE,
                    np.concatenate([[0], [p], np.arange(1, p)]) + i * p,
                )
                for i in range(ncells)
            ]

        elif self._export_dict["level"] == "volume":
            assert isinstance(
                self.cross_section, (ExportableCrossSection)
            ), "Volume export is only implemented for Classes derived from ExportableCrossSection."

            # always use 4 here
            # this is only the number of points we use for the projection
            self._export_dict["num_frames"] = 4 * ncells + 1

            # make cells
            p_cs = self.cross_section.vtk_degree
            p_zeta = 3  # polynomial_degree of the cell along the rod. Always 3, this is hardcoded in L2_projection when BernsteinBasis is called!
            # determines also the number of layers per cell (p_zeta + 1 = 4)
            self._export_dict["p_zeta"] = p_zeta

            higher_order_degree_main = [p_cs, p_cs, p_zeta]
            higher_order_degree_flat = [p_cs, p_cs, 1]

            # get infos from cross section
            ppl = self.cross_section.vtk_points_per_layer
            points_per_cell = (p_zeta + 1) * ppl
            VTK_CELL_TYPE, connectivity_main, connectivity_flat = (
                self.cross_section.vtk_connectivity(p_zeta)
            )

            # create cells and higher_order_degrees
            cells = []
            higher_order_degree = []
            # cap at xi=0
            if self._export_dict["hasCap"]:
                offset = -p_zeta * ppl
                cells.append((VTK_CELL_TYPE, connectivity_flat + offset))
                higher_order_degree.append(higher_order_degree_flat)

            # iterate all cells
            offset = ppl if self._export_dict["hasCap"] else 0
            for i in range(ncells):
                this_offset = i * points_per_cell + offset
                cells.extend(
                    [
                        (VTK_CELL_TYPE, connectivity_main + this_offset),
                        (VTK_CELL_TYPE, connectivity_flat + this_offset),
                    ]
                )
                higher_order_degree.extend(
                    [higher_order_degree_main, higher_order_degree_flat]
                )

            # remove cap at xi=1
            if not self._export_dict["hasCap"]:
                cells = cells[:-1]
                higher_order_degree = higher_order_degree[:-1]

            # Note: if there is a closed rod (ring), this should be handled here

            # save for later use
            self._export_dict["higher_order_degrees"] = higher_order_degree
            self._export_dict["cells"] = cells

            ###############################
            # assertion for stress export #
            ###############################
            if self._export_dict["stresses"]:
                # TODO: remove this assertion, when we compute the stresses on element level
                # stresses are evaluated at xis
                # stresses are very likely to jump at xis_e_int
                # therfore, we shouldn't evaluate them at these points
                num = self._export_dict["num_frames"] - 1
                xis = np.linspace(0, 1, num=num)
                xis_e_int = np.linspace(0, 1, self.nelement + 1)[1:-1]
                for xi_e in xis_e_int:
                    assert (
                        xi_e not in xis
                    ), f"xis for fitting may not contain internal element boundaries to represent discontinuities in stresses. \nInternal boundaries at {xis_e_int}, \nxis_fitting={xis}."

            #################################
            # assertion for surface normals #
            #################################
            if self._export_dict["surface_normals"]:
                assert hasattr(self.cross_section, "B_r_PQ")
                assert hasattr(self.cross_section, "B_r_PQ_eta")
                assert isinstance(self.cross_section, CircularCrossSection)
                warn("Implementation of surface normals is experimental!")
                if not self._export_dict["hasCap"]:
                    warn(
                        "It is recommended to set 'hasCap=True' in rod._export_dict when surface normals are exported!"
                    )

        elif self._export_dict["level"] == "None" or self._export_dict["level"] == None:
            self._export_dict["cells"] = []

        # set flag value to True
        self.preprocessed_export = True

    def export(self, sol_i, **kwargs):
        q = sol_i.q

        if not self.preprocessed_export:
            self.preprocess_export()

        # get values that very computed in preprocess into local scope
        ncells = self._export_dict["ncells"]
        continuity = self._export_dict["continuity"]
        level = self._export_dict["level"]
        num = self._export_dict["num_frames"]

        # get frames
        r_OPs, d1s, d2s, d3s = self.frames(q, num=num)

        if level == "centerline + directors":
            #######################################
            # simple export of points and directors
            #######################################
            # fill structs for centerline and directors
            vtk_points = r_OPs.T

            point_data = {
                "d1": d1s.T,
                "d2": d2s.T,
                "d3": d3s.T,
            }
            # TODO: add stresses here?
            # here is a bit more work to do, as the points allo for now only C0 continuity in stresses!
            cell_data = {}

            return vtk_points, self._export_dict["cells"], point_data, cell_data

        elif level == "volume":
            ################################
            # project on cubic Bezier volume
            ################################
            # project directors on cubic C1 bezier curve
            # TODO: I think there is time-saving potential in the L2_projection_Bezier_curve (e.g., the matrix A there is constant, but created and filled everytime)
            r_OP_segments = L2_projection_Bezier_curve(
                r_OPs.T, ncells, case=continuity
            )[2]
            d1_segments = L2_projection_Bezier_curve(d1s.T, ncells, case=continuity)[2]
            d2_segments = L2_projection_Bezier_curve(d2s.T, ncells, case=continuity)[2]
            d3_segments = L2_projection_Bezier_curve(d3s.T, ncells, case=continuity)[2]

            # get characteristic points from the cross-section
            compute_points = self.cross_section.vtk_compute_points(
                r_OP_segments, d2_segments, d3_segments
            )

            # get some values for shortcuts
            ppl = self.cross_section.vtk_points_per_layer
            p_zeta = self._export_dict["p_zeta"]

            ##################
            # compute points #
            ##################
            vtk_points_weights = []
            # cap at xi=0
            if self._export_dict["hasCap"]:
                vtk_points_weights.extend(compute_points(0, 0))

            # iterate all cells
            for i in range(ncells):
                # iterate all layers
                for layer in range(p_zeta + 1):
                    vtk_points_weights.extend(compute_points(i, layer))

            # cap at xi=1
            if self._export_dict["hasCap"]:
                vtk_points_weights.extend(compute_points(-1, -1))

            # points to export is just the R^3 part
            vtk_points_weights = np.array(vtk_points_weights)
            vtk_points = vtk_points_weights[:, :3]

            # the 4th component are the rational weights
            point_data = {
                "RationalWeights": vtk_points_weights[:, 3, None],
            }

            ###################
            # add points data #
            ###################
            # streses
            if self._export_dict["stresses"]:
                # TODO: do this on element basis when eval_stresses accepts el as argument
                # This needs than a general rewriting!
                num = self._export_dict["num_frames"] - 1
                xis = np.linspace(0, 1, num=num)
                B_ns = np.zeros([3, num])
                B_ms = np.zeros([3, num])

                t = sol_i.t
                la_c = sol_i.la_c
                la_g = sol_i.la_g
                for j in range(num):
                    B_ns[:, j], B_ms[:, j] = self.eval_stresses(
                        t, q, la_c, la_g, xis[j], el=None
                    )

                # project contact forces and moments on cubic C1 bezier curve
                B_n_segments = L2_projection_Bezier_curve(
                    B_ns.T,
                    ncells,
                    case="C-1",
                )[2]
                B_m_segments = L2_projection_Bezier_curve(
                    B_ms.T,
                    ncells,
                    case="C-1",
                )[2]

                # fill lists
                vtk_B_n = []
                vtk_B_m = []
                # cap at xi=0
                if self._export_dict["hasCap"]:
                    vtk_B_n.extend(np.repeat([B_n_segments[0, 0]], ppl, axis=0))
                    vtk_B_m.extend(np.repeat([B_m_segments[0, 0]], ppl, axis=0))

                # iterate all cells
                for i in range(ncells):
                    # iterate all layers
                    for layer in range(p_zeta + 1):
                        vtk_B_n.extend(np.repeat([B_n_segments[i, layer]], ppl, axis=0))
                        vtk_B_m.extend(np.repeat([B_m_segments[i, layer]], ppl, axis=0))

                # cap at xi=1
                if self._export_dict["hasCap"]:
                    vtk_B_n.extend(np.repeat([B_n_segments[-1, -1]], ppl, axis=0))
                    vtk_B_m.extend(np.repeat([B_m_segments[-1, -1]], ppl, axis=0))

                # add them to dictionary with point data
                point_data["B_n"] = vtk_B_n
                point_data["B_m"] = vtk_B_m

            # volume directors
            if self._export_dict["volume_directors"]:
                vtk_d1 = []
                vtk_d2 = []
                vtk_d3 = []
                # cap at xi=0
                if self._export_dict["hasCap"]:
                    vtk_d1.extend(np.repeat([d1_segments[0, 0]], ppl, axis=0))
                    vtk_d2.extend(np.repeat([d2_segments[0, 0]], ppl, axis=0))
                    vtk_d3.extend(np.repeat([d3_segments[0, 0]], ppl, axis=0))

                # iterate all cells
                for i in range(ncells):
                    # iterate all layers
                    for layer in range(p_zeta + 1):
                        vtk_d1.extend(np.repeat([d1_segments[i, layer]], ppl, axis=0))
                        vtk_d2.extend(np.repeat([d2_segments[i, layer]], ppl, axis=0))
                        vtk_d3.extend(np.repeat([d3_segments[i, layer]], ppl, axis=0))

                # cap at xi=1
                if self._export_dict["hasCap"]:
                    vtk_d1.extend(np.repeat([d1_segments[-1, -1]], ppl, axis=0))
                    vtk_d2.extend(np.repeat([d2_segments[-1, -1]], ppl, axis=0))
                    vtk_d3.extend(np.repeat([d3_segments[-1, -1]], ppl, axis=0))

                # add them to dictionary with point data
                point_data["d1"] = vtk_d1
                point_data["d2"] = vtk_d2
                point_data["d3"] = vtk_d3

            # surface normals
            if self._export_dict["surface_normals"]:
                # eta values of the points in each layer
                etas = self.cross_section.point_etas
                vtk_surface_normals = []
                # cap at xi=0
                if self._export_dict["hasCap"]:
                    vtk_surface_normals.extend(
                        np.repeat([-d1_segments[0, 0]], ppl, axis=0)
                    )

                # iterate all cells
                for i in range(ncells):
                    # iterate all layers
                    for layer in range(p_zeta + 1):
                        el = i
                        xi = (i + layer / p_zeta) / ncells
                        # TODO: do this cleaner (computation of el)
                        # TODO: remove the workaround when we can call the basis functions with element number
                        if i != ncells and layer == p_zeta:
                            xi -= 1e-9
                        for point in range(ppl):
                            vtk_surface_normals.append(
                                self.surface_normal(q, xi, etas[point], el)
                            )

                # cap at xi=1
                if self._export_dict["hasCap"]:
                    vtk_surface_normals.extend(
                        np.repeat([d1_segments[-1, -1]], ppl, axis=0)
                    )

                # add them to dictionary with point data
                point_data["surface_normal"] = vtk_surface_normals

            #################
            # add cell data #
            #################
            cell_data = {
                "HigherOrderDegrees": self._export_dict["higher_order_degrees"],
            }

        elif level == "None" or level == None:
            vtk_points = []
            point_data = {}
            cell_data = {}

        return vtk_points, self._export_dict["cells"], point_data, cell_data
