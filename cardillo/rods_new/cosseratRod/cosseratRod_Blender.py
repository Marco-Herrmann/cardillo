import numpy as np
from pathlib import Path
from pygltflib import (
    Mesh,
    Primitive,
    Node,
    Skin,
    GLTF2,
    Buffer,
    Scene,
    AnimationSampler,
    AnimationChannel,
    AnimationChannelTarget,
    Animation,
)

from cardillo.visualization.glTF_export import (
    BufferBuilder,
    cardillo_to_gltf_trans,
    cardillo_to_gltf_rot,
)


class RodBlenderExport:
    def create_object(self, xis, buf):
        verts, indices, joints, weights = self.cross_section.create_mesh(xis)

        # inverse binding matrices (TODO: what is that)
        eye4 = np.eye(4, dtype=np.float32)
        ibm = np.array([eye4 for _ in range(len(xis))])

        pos_acc = buf.add(verts, 5126, "VEC3", set_bounds=True)
        idx_acc = buf.add(indices, 5125, "SCALAR")
        joint_acc = buf.add(joints, 5123, "VEC4")
        weight_acc = buf.add(weights, 5126, "VEC4")
        ibm_acc = buf.add(ibm.reshape(-1, 16), 5126, "MAT4")

        mesh = Mesh(
            name=f"{self.name}_mesh",
            primitives=[
                Primitive(
                    attributes={
                        "POSITION": pos_acc,
                        "JOINTS_0": joint_acc,
                        "WEIGHTS_0": weight_acc,
                    },
                    indices=idx_acc,
                )
            ],
        )

        nodes = [Node(name=f"bone_{i}_obj") for i in range(len(xis))]
        mesh_node = Node(
            name=f"{self.name}_obj__invisible",
            mesh=0,
            skin=0,
            children=list(range(len(xis))),
        )
        nodes.append(mesh_node)

        skin = Skin(
            name=f"{self.name}_skin",
            joints=list(range(len(xis))),
            inverseBindMatrices=ibm_acc,
        )

        return [mesh], nodes, [skin]

    def export_blender(self, path, solution):
        xis, data = self.kinematics._export_nodes(solution)

        filename = Path(path) / f"{self.name}.glb"

        assert (
            data.shape[2] == 7
        ), "Expected last dimension of data to be 7 (3 for translation + 4 for rotation)."

        buf = BufferBuilder()
        t_acc = buf.add(solution.t.astype(np.float32), 5126, "SCALAR")

        meshes, nodes, skins = self.create_object(xis, buf)

        samplers = []
        channels = []
        for i in range(len(xis)):
            trans = cardillo_to_gltf_trans(data[:, i, :3])
            rot = cardillo_to_gltf_rot(data[:, i, 3:])
            trans_acc = buf.add(trans, 5126, "VEC3")
            rot_acc = buf.add(rot, 5126, "VEC4")

            # Translation sampler
            samplers.append(
                AnimationSampler(input=t_acc, output=trans_acc, interpolation="LINEAR")
            )
            channels.append(
                AnimationChannel(
                    sampler=len(samplers) - 1,
                    target=AnimationChannelTarget(node=i, path="translation"),
                )
            )

            # Rotation sampler
            samplers.append(
                AnimationSampler(input=t_acc, output=rot_acc, interpolation="LINEAR")
            )
            channels.append(
                AnimationChannel(
                    sampler=len(samplers) - 1,
                    target=AnimationChannelTarget(node=i, path="rotation"),
                )
            )

        anim = Animation(samplers=samplers, channels=channels)

        gltf = GLTF2(
            buffers=[Buffer(byteLength=len(buf.data))],
            bufferViews=buf.views,
            accessors=buf.accessors,
            meshes=meshes,
            nodes=nodes,
            skins=skins,
            animations=[anim],
            scenes=[Scene(nodes=[len(nodes) - 1])],
            scene=0,
        )

        gltf.set_binary_blob(buf.data)
        gltf.save_binary(filename)

    def export_blender_modes(self, path, solution):
        xis, data, delta = self.kinematics._export_nodes_modes(solution)

        filename = Path(path) / f"{self.name}.glb"

        assert (
            data.shape[1] == 7
        ), "Expected last dimension of data to be 7 (3 for translation + 4 for rotation)."

        buf = BufferBuilder()
        meshes, nodes, skins = self.create_object(xis, buf)
        node_rod = nodes[-1]
        node_rod.children = []

        for i, node_obj in enumerate(nodes[: len(xis)]):
            node_root = Node(
                name=node_obj.name.replace("_obj", "_root"),
                extras={
                    "r_OP0": data[i, :3].tolist(),
                    "P_IB0": data[i, 3:].tolist(),
                    "omegas": solution.omegas.tolist(),
                    "Delta_r": delta[i, :3].T.tolist(),
                    "B_Delta_phi": delta[i, 3:].T.tolist(),
                },
                children=[i],
            )
            nodes.append(node_root)
            node_rod.children.append(len(nodes) - 1)

        gltf = GLTF2(
            buffers=[Buffer(byteLength=len(buf.data))],
            bufferViews=buf.views,
            accessors=buf.accessors,
            meshes=meshes,
            nodes=nodes,
            skins=skins,
            scenes=[Scene(nodes=[len(xis)])],
            scene=0,
        )

        gltf.set_binary_blob(buf.data)
        gltf.save_binary(filename)
