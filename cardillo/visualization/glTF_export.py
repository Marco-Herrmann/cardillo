import numpy as np
from pygltflib import *


class BufferBuilder:
    def __init__(self):
        self.data = bytearray()
        self.views = []
        self.accessors = []

    def add(self, array, componentType, type_str):
        raw = array.tobytes()
        offset = len(self.data)

        self.data.extend(raw)

        # 4-byte alignment
        while len(self.data) % 4 != 0:
            self.data.append(0)

        view_id = len(self.views)
        self.views.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(raw)))

        accessor_id = len(self.accessors)
        self.accessors.append(
            Accessor(
                bufferView=view_id,
                componentType=componentType,
                count=len(array),
                type=type_str,
            )
        )

        return accessor_id


def cardillo_to_gltf(data):
    trans = data[:, 0:3].astype(np.float32)
    rot = data[:, 3:7].astype(np.float32)

    # normalize quaternions
    rot = rot / np.linalg.norm(rot, axis=1, keepdims=True)

    # cardillo:
    #   x: right,
    #   y: forward,
    #   z: up,
    #   P = (p0, p),
    # glTF:
    #   x: right,
    #   y: up,
    #   z: backward,
    #   P = (p, p0),

    # trans: y <- z and z <- -y
    trans = np.stack([trans[:, 0], trans[:, 2], -trans[:, 1]], axis=1)

    # rot: y <- z and z <- -y
    rot = np.stack([rot[:, 1], rot[:, 3], -rot[:, 2], rot[:, 0]], axis=-1)
    return trans, rot
