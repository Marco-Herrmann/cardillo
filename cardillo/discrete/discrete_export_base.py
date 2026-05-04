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
from cardillo.math.rotations import Exp_SO3_quat
from cardillo.math.SmallestRotation import smallest_rotation_quaternion


def make_glTF(path, name, t, r_OP, v_P=None, P_IB=None, B_Omega=None, mesh=None):
    # TODO: get rid of os
    filename = os.path.join(path, f"{name}.glb")

    buf = BufferBuilder()
    nodes = []
    # create an empty
    if mesh is None:
        nodes.append(Node(name=f"{name}_obj"))
        mesh_gltf = []
    else:
        # create a mesh node
        verts = cardillo_to_gltf_trans(mesh.vertices)
        faces = mesh.faces.astype(np.uint32)

        pos_acc = buf.add(verts, 5126, "VEC3")
        idx_acc = buf.add(faces.reshape(-1), 5125, "SCALAR")

        prim = Primitive(attributes={"POSITION": pos_acc}, indices=idx_acc)

        nodes.append(Node(name=f"{name}_obj", mesh=0))
        mesh_gltf = [Mesh(primitives=[prim], name=f"{name}_mesh")]

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
                sampler=len(samplers) - 1,
                target=AnimationChannelTarget(node=0, path="rotation"),
            )
        )

    if v_P is not None:
        nodes.append(Node(name=f"{name}_v"))
        nodes[0].children.append(len(nodes) - 1)

        # compute B_v_P, as we parent it w.r.t. the object
        if P_IB is not None:
            B_v_P = np.einsum("ikj,ik->ij", Exp_SO3_quat(P_IB), v_P)
        else:
            B_v_P = v_P

        # find the quaternion that rotates the +z of the node to the direction of v_P
        # TODO: vectorize
        P = np.array([smallest_rotation_quaternion(B_v_Pi, i=2)[0] for B_v_Pi in B_v_P])
        P_rot = cardillo_to_gltf_rot(P)
        P_acc = buf.add(P_rot, 5126, "VEC4")

        samplers.append(
            AnimationSampler(input=t_acc, output=P_acc, interpolation="LINEAR")
        )
        channels.append(
            AnimationChannel(
                sampler=len(samplers) - 1,
                target=AnimationChannelTarget(node=len(nodes) - 1, path="rotation"),
            )
        )

        # set the scale of the node to the magnitude of v_P
        B_Omeg_mag = np.linalg.norm(v_P, axis=1).astype(np.float32)
        v_P_acc = buf.add(np.stack([B_Omeg_mag] * 3, axis=1), 5126, "VEC3")
        samplers.append(
            AnimationSampler(input=t_acc, output=v_P_acc, interpolation="LINEAR")
        )
        channels.append(
            AnimationChannel(
                sampler=len(samplers) - 1,
                target=AnimationChannelTarget(node=len(nodes) - 1, path="scale"),
            )
        )

    if B_Omega is not None:
        assert P_IB is not None, "B_Omega export requires P_IB"
        nodes.append(Node(name=f"{name}_Omega"))
        nodes[0].children.append(len(nodes) - 1)

        # find the quaternion that rotates the +z of the node to the direction of v_P
        # TODO: vectorize
        P = np.array(
            [smallest_rotation_quaternion(B_Omegai, i=2)[0] for B_Omegai in B_Omega]
        )
        P_rot = cardillo_to_gltf_rot(P)
        P_acc = buf.add(P_rot, 5126, "VEC4")

        samplers.append(
            AnimationSampler(input=t_acc, output=P_acc, interpolation="LINEAR")
        )
        channels.append(
            AnimationChannel(
                sampler=len(samplers) - 1,
                target=AnimationChannelTarget(node=len(nodes) - 1, path="rotation"),
            )
        )

        # set the scale of the node to the magnitude of B_Omega
        B_Omeg_mag = np.linalg.norm(B_Omega, axis=1).astype(np.float32)
        B_Omeg_acc = buf.add(np.stack([B_Omeg_mag] * 3, axis=1), 5126, "VEC3")
        samplers.append(
            AnimationSampler(input=t_acc, output=B_Omeg_acc, interpolation="LINEAR")
        )
        channels.append(
            AnimationChannel(
                sampler=len(samplers) - 1,
                target=AnimationChannelTarget(node=len(nodes) - 1, path="scale"),
            )
        )

    anim = Animation(samplers=samplers, channels=channels, name=f"{name}_anim")
    gltf = GLTF2(
        buffers=[Buffer(byteLength=len(buf.data))],
        bufferViews=buf.views,
        accessors=buf.accessors,
        nodes=nodes,
        animations=[anim],
        scenes=[Scene(nodes=[0])],
        scene=0,
        meshes=mesh_gltf,
    )

    gltf.set_binary_blob(buf.data)
    gltf.save_binary(filename)
