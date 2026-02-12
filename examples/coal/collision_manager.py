import numpy as np
import coal
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    # create coal shapes
    shape1 = coal.Ellipsoid(0.7, 1.0, 0.8)
    shape2 = coal.Cylinder(0.5, 1.5)
    shape3 = coal.Capsule(0.5, 1.5)

    shape1 = coal.Sphere(2.0)
    shape2 = coal.Sphere(3.0)

    # collision objects
    obj1 = coal.CollisionObject(shape1)
    obj2 = coal.CollisionObject(shape2)
    obj3 = coal.CollisionObject(shape3)

    # shape and object lists
    shapes = [shape1, shape2, shape3]
    objs = [obj1, obj2, obj3]

    shapes = [shape1, shape2]
    objs = [obj1, obj2]

    # create map from shape IDs to objects
    shape_id_to_obj = {shape.id(): obj for shape, obj in zip(shapes, objs)}

    # create broadphase manager
    collision_manager = coal.DynamicAABBTreeCollisionManager()
    for obj in objs:
        collision_manager.registerObject(obj)
    collision_manager.setup()

    # default callback
    callback = coal.CollisionCallBackDefault()
    # TODO: Check other required options for callback
    callback.data.request.num_max_contacts = 100

    # create collision request structure
    crequest = coal.CollisionRequest()
    cdata = coal.CollisionData()

    #######################
    # dummy simulation loop
    #######################
    if True:
        for obj in objs:
            # set positions and orientations
            T = coal.Transform3s()
            T.setTranslation(np.random.rand(3))
            T.setRotation(Rotation.random().as_matrix())
            obj.setTransform(T)

        # update manager
        collision_manager.update()

        # # collect collision pairs
        # collision_size = 100
        # collect = coal.CollisionCallBackCollect(collision_size)

        # collide
        # out = collision_manager.collide(collect)
        out = collision_manager.collide(callback)

        # extract contacts
        # callback.data.result.getContact(0) # single contact for testing
        contacts = callback.data.result.getContacts()
        contacts_list = [c for c in contacts]

        for i, contact in enumerate(contacts):
            # access contact information
            position = contact.pos
            normal = contact.normal
            penetration_depth = contact.penetration_depth

            # access contact shapes
            s1 = contact.o1  # first shape in contact
            s2 = contact.o2  # second shape in contact

            # get corresponding objects:
            # This can be used to access the underlying cardillo body to
            # get the Jacobians, etc.
            o1 = shape_id_to_obj.get(s1.id())
            o2 = shape_id_to_obj.get(s2.id())

            r_OP1 = o1.getTransform().getTranslation()
            r_OP2 = o2.getTransform().getTranslation()

            r_OQ1 = contact.getNearestPoint1()
            r_OQ2 = contact.getNearestPoint2()

            print(f"contact {i}:")

            print(f" - position: {position}")
            print(f" - normal: {normal}")
            print(f" - penetration depth: {penetration_depth}")

            print(f" - shape1: {s1}")
            print(f" - object1: {o1}")

            print(f" - shape2: {s2}")
            print(f" - object2: {o2}")
