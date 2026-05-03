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
