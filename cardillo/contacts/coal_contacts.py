from cachetools import LRUCache, cachedmethod
from cachetools.keys import hashkey
import numpy as np
import coal
from warnings import warn

from cardillo.math.algebra import cross3, norm, e3
from cardillo.math.prox import Sphere

# TODO: get other primitives from meshed
# https://github.com/coal-library/coal/blob/devel/src/shape/geometric_shapes.cpp
# https://github.com/coal-library/coal/blob/devel/include/coal/shape/geometric_shapes.h#L386
# https://docs.ros.org/en/iron/p/coal/generated/classcoal_1_1ShapeBase.html#exhale-class-classcoal-1-1shapebase


def CoalBox(dimensions):
    return coal.Box(*dimensions)


def CoalSphere(radius):
    return coal.Sphere(radius)


def CoalEllipsoid(dimensions):
    # TODO: Test implementation
    return coal.Ellipsoid(*dimensions)


def CoalCapsule(radius, height):
    return coal.Capsule(radius, height)


def CoalCone(radius, height):
    # TODO: coalCone goes from z=-height/2 to z=height/2
    # while in cardillo it goes from z=0 to z=height
    # however, the COM is somewhere between the base surface and the middle
    return coal.Cone(radius, height)


def CoalCylinder(radius, height):
    return coal.Cylinder(radius, height)


# TODO: how to name it: halfspace or plane
# in cardillo, we call it plane, but coal halfspace is oriented, while plane isn't
# def CoalPlane(normal=e3, d=0.0):
def CoalHalfspace(normal=e3, d=0.0):
    # default in upwards direction
    return coal.Halfspace(normal, d)


def CoalTriangleP(): ...


class ContactObject:
    def __init__(
        self,
        subsystem,
        shape,
        xi=None,
        B_r_CP=np.zeros(3, dtype=float),
        A_BJ=np.eye(3, dtype=float),
        name="ContactObject",
    ):
        self.subsystem = subsystem
        self.shape = shape
        self.xi = xi
        self.B_r_CP = B_r_CP
        self.A_BJ = A_BJ
        self.name = name

        self.obj = coal.CollisionObject(self.shape)

    def assembler_callback(self):
        local_qDOF = self.subsystem.local_qDOF_P(self.xi)
        local_uDOF = self.subsystem.local_uDOF_P(self.xi)
        self.qDOF = self.subsystem.qDOF[local_qDOF]
        self.uDOF = self.subsystem.uDOF[local_uDOF]

    def update_coal_transform(self, t, q):
        r_OP = self.subsystem.r_OP(t, q, xi=self.xi, B_r_CP=self.B_r_CP)
        A_IJ = self.subsystem.A_IB(t, q, xi=self.xi) @ self.A_BJ
        transform = coal.Transform3s()
        transform.setTranslation(r_OP)
        transform.setRotation(A_IJ)
        self.obj.setTransform(transform)

    def get_contact_information(self, r_OP, t, q, u):
        r_OC = self.subsystem.r_OP(t, q, xi=self.xi)
        A_IB = self.subsystem.A_IB(t, q, xi=self.xi)
        B_r_CP = A_IB.T @ (r_OP - r_OC)

        J_P = self.subsystem.J_P(t, q, xi=self.xi, B_r_CP=B_r_CP)
        v_P = self.subsystem.v_P(t, q, u, xi=self.xi, B_r_CP=B_r_CP)

        return J_P, v_P

    def contact_basis(self, t, q, n):
        A_IB = self.subsystem.A_IB(t, q, xi=self.xi)

        vs = np.cross(n, A_IB, axisb=0)
        norm_vs = np.linalg.norm(vs, axis=-1)
        # use axis with best angle
        axis = np.argmax(norm_vs)
        t1 = vs[axis] / norm_vs[axis]
        w = cross3(n, t1)
        t2 = w / norm(w)
        return np.vstack((t1, t2, n)).T


class ContactCollection:
    def __init__(
        self,
        contactObjects: list[ContactObject],
        mu=0.0,
        e_N=None,
        e_F=None,
    ):
        self.contactObjects = contactObjects
        self.nobjects = len(self.contactObjects)

        # create map from obj-id pair to la_NDOF
        pairs = [
            frozenset((self.contactObjects[i], self.contactObjects[j]))
            for i in range(self.nobjects)
            for j in range(i + 1, self.nobjects)
        ]
        self.pair_to_la_NFDOF = {
            pair: (i, [2 * i, 2 * i + 1]) for i, pair in enumerate(pairs)
        }

        # create map from shape IDs to objects
        self.shape_id_to_obj = {obj.shape.id(): obj for obj in self.contactObjects}

        # create broadphase manager
        self.collision_manager = coal.DynamicAABBTreeCollisionManager()
        for obj in self.contactObjects:
            self.collision_manager.registerObject(obj.obj)
        self.collision_manager.setup()

        # default callback
        self.callback = coal.CollisionCallBackDefault()
        # TODO: Check other required options for callback
        self.callback.data.request.num_max_contacts = 100

        # create collision request structure
        self.crequest = coal.CollisionRequest()
        self.cdata = coal.CollisionData()

        # compute number of contacts and initialize restitution coefficients
        self.nla_N = len(pairs)
        self.e_N = np.zeros(self.nla_N) if e_N is None else e_N * np.ones(self.nla_N)

        if mu > 0:
            self.active_friction = True
            self.nla_F = 2 * self.nla_N
            self.gamma_F = self._gamma_F
            self.gamma_F_q = self._gamma_F_q
            self.e_F = (
                np.zeros(self.nla_F) if e_F is None else e_F * np.ones(self.nla_F)
            )

            # fmt: off
            self.friction_laws = [
                ([i], [2*i, 2*i+1], Sphere(mu)) for i in range(self.nla_N) # Coulomb
            ]
            # fmt: on
        else:
            self.active_friction = False

        self.chef_cache = LRUCache(maxsize=1)

    def change_contact_property(self, *args, **kwargs):
        # TODO: move then a lof of things from __init__ to assembler callback
        raise NotImplementedError

    def assembler_callback(self):
        qDOF = 0
        uDOF = 0
        for obj in self.contactObjects:
            obj.assembler_callback()

            qDOF = np.max([qDOF, *obj.qDOF])
            uDOF = np.max([uDOF, *obj.uDOF])

        self.qDOF = np.arange(qDOF + 1)
        self.uDOF = np.arange(uDOF + 1)

        self.my_nq = len(self.qDOF)
        self.my_nu = len(self.uDOF)

    @cachedmethod(
        lambda self: self.chef_cache, key=lambda self, t, q, u: hashkey(t, *q, *u)
    )
    def chef(self, t, q, u):
        for obj in self.contactObjects:
            obj.update_coal_transform(t, q[obj.qDOF])

        # update manager
        self.collision_manager.update()

        # collide
        self.collision_manager.collide(self.callback)

        # extract contacts
        contacts = self.callback.data.result.getContacts()

        # create arrays to return
        g_N = np.ones(self.nla_N)
        g_N_dot = np.ones(self.nla_N)
        # TODO: use coo-matrix
        W_N = np.zeros((self.my_nu, self.nla_N))

        if self.active_friction:
            gamma_F = np.zeros(self.nla_F)
            # TODO: use coo-matrix
            W_F = np.zeros((self.my_nu, self.nla_F))

        for i, contact in enumerate(contacts):
            # access contact information
            r_OP = contact.pos
            r_OP1 = contact.getNearestPoint1()
            r_OP2 = contact.getNearestPoint2()
            normal = contact.normal
            penetration_depth = contact.penetration_depth

            # access contact shapes
            shape1 = contact.o1  # first shape in contact
            shape2 = contact.o2  # second shape in contact

            # get corresponding objects:
            # This can be used to access the underlying cardillo body to
            # get the Jacobians, etc.
            obj1 = self.shape_id_to_obj.get(shape1.id())
            obj2 = self.shape_id_to_obj.get(shape2.id())

            la_NDOF, la_FDOF = self.pair_to_la_NFDOF.get(frozenset((obj1, obj2)))

            # compute jacobians and velcoities
            J_P1, v_P1 = obj1.get_contact_information(
                r_OP1, t, q[obj1.qDOF], u[obj1.uDOF]
            )
            J_P2, v_P2 = obj2.get_contact_information(
                r_OP2, t, q[obj2.qDOF], u[obj2.uDOF]
            )

            g_N[la_NDOF] = penetration_depth

            v_P1P2 = v_P2 - v_P1
            g_N_dot[la_NDOF] = normal @ v_P1P2

            W_N[obj1.uDOF, la_NDOF] = -normal @ J_P1
            W_N[obj2.uDOF, la_NDOF] = normal @ J_P2

            # friction
            if self.active_friction:
                A_IC = obj1.contact_basis(t, q[obj1.qDOF], normal)
                t1 = A_IC[:, 0]
                t2 = A_IC[:, 1]

                gamma_F[la_FDOF[0]] = t1 @ v_P1P2
                gamma_F[la_FDOF[1]] = t2 @ v_P1P2

                W_F[obj1.uDOF, la_FDOF[0]] = -t1 @ J_P1
                W_F[obj2.uDOF, la_FDOF[0]] = t1 @ J_P2
                W_F[obj1.uDOF, la_FDOF[1]] = -t2 @ J_P1
                W_F[obj2.uDOF, la_FDOF[1]] = t2 @ J_P2

        return g_N, g_N_dot, W_N, gamma_F, W_F

    ################
    # normal contact
    ################
    def g_N(self, t, q):
        return self.chef(t, q, np.zeros(len(self.uDOF)))[0]

    def g_N_q(self, t, q):
        raise NotImplementedError

    def g_N_dot(self, t, q, u):
        return self.chef(t, q, u)[1]

    def g_N_dot_q(self, t, q, u):
        raise NotImplementedError

    def g_N_dot_u(self, t, q):
        raise NotImplementedError

    def W_N(self, t, q):
        return self.chef(t, q, np.zeros(len(self.uDOF)))[2]

    def g_N_ddot(self, t, q, u, u_dot):
        # prevent error for consistent initial conditions
        warn("ContactCollection: g_N_ddot not implemented yet!")

    def Wla_N_q(self, t, q, la_N):
        raise NotImplementedError

    ##########
    # friction
    ##########
    def _gamma_F(self, t, q, u):
        return self.chef(t, q, u)[3]

    def _gamma_F_q(self, t, q, u):
        raise NotImplementedError

    def gamma_F_u(self, t, q):
        raise NotImplementedError

    def W_F(self, t, q):
        return self.chef(t, q, np.zeros(len(self.uDOF)))[4]

    def gamma_F_dot(self, t, q, u, u_dot):
        # prevent error for consistent initial conditions
        warn("ContactCollection: gamma_F_dot not implemented yet!")

    def Wla_F_q(self, t, q, la_F):
        raise NotImplementedError
