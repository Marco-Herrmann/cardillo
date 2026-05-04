import numpy as np
import os
from pygltflib import (
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    Buffer,
    GLTF2,
    Mesh,
    Node,
    Primitive,
    Scene,
)
from warnings import warn

from cardillo.visualization.glTF_export import (
    BufferBuilder,
    cardillo_to_gltf_trans,
    cardillo_to_gltf_rot,
)


def make_glTF(path, name, t, r_OP, v_P=None, P_IB=None, B_Omega=None, mesh=None):
    # TODO: get rid of os
    filename = os.path.join(path, f"{name}.glb")

    buf = BufferBuilder()
    # create an empty
    if mesh is None:
        node = Node(name=f"{name}_obj")
        mesh_gltf = []
    else:
        # create a mesh node
        verts = cardillo_to_gltf_trans(mesh.vertices)
        faces = mesh.faces.astype(np.uint32)

        pos_acc = buf.add(verts, 5126, "VEC3")
        idx_acc = buf.add(faces.reshape(-1), 5125, "SCALAR")

        prim = Primitive(attributes={"POSITION": pos_acc}, indices=idx_acc)

        node = Node(name=f"{name}_obj", mesh=0)
        mesh_gltf = [Mesh(primitives=[prim])]

    t_acc = buf.add(t.astype(np.float32), 5126, "SCALAR")

    # translation
    trans = cardillo_to_gltf_trans(r_OP)
    trans_acc = buf.add(trans, 5126, "VEC3")
    samplers = [
        AnimationSampler(input=t_acc, output=trans_acc, interpolation="LINEAR"),
    ]
    channels = [
        AnimationChannel(
            sampler=0,
            target=AnimationChannelTarget(node=0, path="translation"),
        ),
    ]

    # orientation
    if P_IB is not None:
        rot = cardillo_to_gltf_rot(P_IB)
        rot_acc = buf.add(rot, 5126, "VEC4")

        samplers.append(
            AnimationSampler(input=t_acc, output=rot_acc, interpolation="LINEAR")
        )
        channels.append(
            AnimationChannel(
                sampler=1,
                target=AnimationChannelTarget(node=0, path="rotation"),
            )
        )

    # TODO: implement v_P and B_Omega export
    if v_P is not None:
        warn("v_P export not implemented yet")
    if B_Omega is not None:
        assert P_IB is not None, "B_Omega export requires P_IB"
        warn("B_Omega export not implemented yet")

    anim = Animation(samplers=samplers, channels=channels)
    gltf = GLTF2(
        buffers=[Buffer(byteLength=len(buf.data))],
        bufferViews=buf.views,
        accessors=buf.accessors,
        nodes=[node],
        animations=[anim],
        scenes=[Scene(nodes=[0])],
        scene=0,
        meshes=mesh_gltf,
    )

    gltf.set_binary_blob(buf.data)
    gltf.save_binary(filename)
