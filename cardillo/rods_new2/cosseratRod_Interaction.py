import numpy as np


class CosseratRod_Interaction:
    def local_qDOF_P(self, xi):
        return self.get_interaction_point(xi).get("qDOF")

    def local_uDOF_P(self, xi):
        return self.get_interaction_point(xi).get("uDOF")

    def get_interaction_point(self, xi):
        # TODO: call this from constraints, see Tianxiang Marker, so that it is not called from postprocessing r_OP, etc. calls
        # TODO: check that it is always done using this function, never access interaction points directly!
        if not (xi in self.interaction_points.keys()):
            nq_node = self.kinematics.nq_node
            nu_node = self.velocity.nu_node
            if (node_number := self.node_number(xi)) is not False:
                nnodes = 1
                qDOF = np.arange(nq_node) + nq_node * node_number
                uDOF = np.arange(nu_node) + nu_node * node_number

                Nq = np.eye(nq_node)
                Nu = np.eye(nu_node)
                N = np.array([1.0])
            else:
                el = self.element_number(xi)
                p = self._polynomial_degree
                nnodes = p + 1

                N = self.N_element(xi, el)
                Nq = np.zeros((nq_node, nq_node * nnodes))
                rows = np.arange(nq_node)[:, None]
                cols = rows + nq_node * np.arange(nnodes)
                Nq[rows, cols] = N

                Nu = np.zeros((nu_node, nu_node * nnodes))
                rows = np.arange(nu_node)[:, None]
                cols = rows + nu_node * np.arange(nnodes)
                Nu[rows, cols] = N

                # elDOF
                # TODO: clean up
                if self._IGA:
                    s = p - self.continuity
                    start = s * el
                    end = s * el + p + 1
                else:
                    start = p * el
                    end = p * (el + 1) + 1
                qDOF = np.arange(nq_node * start, nq_node * end)
                uDOF = np.arange(nu_node * start, nu_node * end)

            self.interaction_points[xi] = dict(
                nnodes=nnodes,
                qDOF=qDOF,
                uDOF=uDOF,
                N=N,
                Nq=Nq,
                Nu=Nu,
                # r_q=Nq[:3],
                # P_q=Nq[3:],
                # J_C=Nu[:3],
                # B_J_R=Nu[3:],
                zero_3_nqi=np.zeros((3, nq_node * nnodes), dtype=float),
                zero_3_nui=np.zeros((3, nu_node * nnodes), dtype=float),
                zero_3_nui_nui=np.zeros(
                    (3, nu_node * nnodes, nu_node * nnodes), dtype=float
                ),
                zero_3_nui_nqi=np.zeros(
                    (3, nu_node * nnodes, nq_node * nnodes), dtype=float
                ),
            )
        return self.interaction_points[xi]
