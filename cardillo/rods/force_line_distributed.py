from warnings import warn
import numpy as np
from cardillo.math.algebra import norm, cross3, ax2skew
from cardillo.rods.boostedCosseratRod import CosseratRod_PetrovGalerkin


class Force_line_distributed:
    def __init__(self, force, rod, p_ext=0):
        r"""Line distributed dead load for rods

        Parameters
        ----------
        force : np.ndarray (3,)
            Force w.r.t. inertial I-basis as a callable function in time t and
            rod position xi.
        rod : CosseratRod
            Cosserat rod from Cardillo.

        """
        if not callable(force):
            self.force = lambda t, xi: force
        else:
            self.force = force
        self.rod = rod

        if hasattr(rod, "polynomial_degree"):
            # boosted implementation
            p_rod = rod.polynomial_degree
            mesh = rod.mesh_kin
        elif hasattr(rod, "polynomial_degree_r"):
            # default implementation
            p_rod = rod.polynomial_degree_r
            mesh = rod.mesh_r
        else:
            raise RuntimeError("Could not find rod's polynomial degree.")

        self.nquadrature = int(np.ceil((p_rod + p_ext + 1) / 2))
        self.qp, self.qw = mesh.quadrature_points(self.nquadrature)

        if isinstance(rod, CosseratRod_PetrovGalerkin):
            self.h = self.h_new

            # TODO: think of a convenient way to get all the stuff
            Nq_, Nu_ = mesh.shape_functions_matrix(self.nquadrature, 1)
            Nq, Nq_xi = Nq_
            self.Nu = Nu_[0][:, :, :3, : self.rod.nu_element_r]

            self.J = np.zeros_like(self.qp)
            for el in range(rod.nelement):
                qe = self.rod.Q[self.rod.elDOF[el]]
                for i in range(self.nquadrature):
                    qpi = self.qp[el, i]
                    self.J[el, i] = rod.compute_J(qpi, Nq[el, i], Nq_xi[el, i], qe)

        else:
            self.h = self.h_old

    def assembler_callback(self):
        self.qDOF = self.rod.qDOF
        self.uDOF = self.rod.uDOF

    ##################
    # potential energy
    ##################
    def E_pot(self, t, q):
        E_pot = 0
        for el in range(self.rod.nelement):
            qe = q[self.elDOF[el]]
            E_pot += self.E_pot_el(t, qe, el)
        return E_pot

    def E_pot_el(self, t, qe, el):
        # TODO: nullify with initial configuration q0
        E_pot_el = 0.0

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            Ji = self.rod.J[el, i]

            # interpolate centerline position
            r_OC = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_r):
                r_OC += self.N_r[el, i, node] * qe[self.nodalDOF_element_r[node]]

            E_pot_el -= (r_OC @ self.force(t, qpi)) * Ji * qwi

        return E_pot_el

    #####################
    # equations of motion
    #####################
    def h_old(self, t, q, u):
        h = np.zeros(self.rod.nu, dtype=np.common_type(q, u))
        for el in range(self.rod.nelement):
            h[self.rod.elDOF_u[el]] += self.h_el_old(t, el)
        return h

    def h_el_old(self, t, el):
        he = np.zeros(self.rod.nu_element, dtype=float)

        for i in range(self.rod.nquadrature):
            # extract reference state variables
            qpi = self.rod.qp[el, i]
            qwi = self.rod.qw[el, i]
            Ji = self.rod.J[el, i]

            # compute local force vector
            he_qp = self.force(t, qpi) * Ji * qwi

            # multiply local force vector with variation of centerline
            for node in range(self.rod.nnodes_element_r):
                he[self.rod.nodalDOF_element_r[node]] += (
                    self.rod.N_r[el, i, node] * he_qp
                )

        return he

    def h_new(self, t, q, u):
        h = np.zeros(self.rod.nu, dtype=np.common_type(q, u))
        for el in range(self.rod.nelement):
            elDOF_u = self.rod.elDOF_u[el][: self.rod.nu_element_r]
            h[elDOF_u] += self.h_el_new(t, el)
        return h

    def h_el_new(self, t, el):
        h_el = np.zeros(self.rod.nu_element_r, dtype=float)

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # compute local force vector
            he_qp = self.force(t, qpi) * Ji * qwi
            h_el += self.Nu[el, i].T @ he_qp

        return h_el

    def h_q(self, t, q, u):
        return None
