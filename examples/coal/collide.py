import numpy as np
import coal
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    # Create coal shapes
    shape1 = coal.Ellipsoid(0.7, 1.0, 0.8)
    shape2 = coal.Cylinder(0.5, 1.5)

    shape1.name = "Ellipsoid1"
    shape2.name = "Cylinder1"

    # Collision objects
    obj1 = coal.CollisionObject(shape1)
    obj2 = coal.CollisionObject(shape2)

    # shape and object lists
    shapes = [shape1, shape2]
    objs = [obj1, obj2]
    # names = ['obj1', 'obj2']

    shape_pairs = []
    n = len(shapes)
    for i in range(n):
        for j in range(i + 1, n):
            shape_pairs.append((shapes[i], shapes[j]))
    print(f"collision pairs: {shape_pairs}")

    # Create map from geometry IDs to objects
    shape_id_to_obj = {id(shape): obj for shape, obj in zip(shapes, objs)}

    #######################
    # dummy simulation loop
    #######################
    if True:
        # set positions and orientations
        T1 = coal.Transform3s()
        T1.setTranslation(np.random.rand(3))
        T1.setRotation(Rotation.random().as_matrix())

        T2 = coal.Transform3s()
        T2.setTranslation(np.random.rand(3))
        T2.setRotation(Rotation.random().as_matrix())

        # Define collision requests and results
        col_req = coal.CollisionRequest()
        col_res = coal.CollisionResult()

        # Collision calls
        for pair in shape_pairs:
            coal.collide(pair[0], T1, pair[1], T2, col_req, col_res)

            # Accessing the collision result once it has been populated
            print("Is collision? ", {col_res.isCollision()})
            if col_res.isCollision():
                contact: coal.Contact = col_res.getContact(0)

                position = contact.pos
                normal = contact.normal
                penetration_depth = contact.penetration_depth
                # s1 = contact.o1 # first shape (not the original one, but a copy inside coal)
                # s2 = contact.o2 # second shape (not the original one, but a copy inside coal)

            # Before running another collision call, it is important to clear the old one
            col_res.clear()
