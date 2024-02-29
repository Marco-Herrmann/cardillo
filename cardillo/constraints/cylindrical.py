import numpy as np
from cardillo.constraints._base import ProjectedPositionOrientationBase


class Cylindrical(ProjectedPositionOrientationBase):
    def __init__(
        self,
        subsystem1,
        subsystem2,
        axis,
        r_OB0=None,
        A_IB0=None,
        frame_ID1=None,
        frame_ID2=None,
    ):
        assert axis in (0, 1, 2)

        # remove free axis
        constrained_axes_displacement = np.delete((0, 1, 2), axis)

        # project free axis onto both constrained axes
        projection_pairs_rotation = [
            (axis, constrained_axes_displacement[0]),
            (axis, constrained_axes_displacement[1]),
        ]

        super().__init__(
            subsystem1,
            subsystem2,
            r_OB0=r_OB0,
            A_IB0=A_IB0,
            constrained_axes_translation=constrained_axes_displacement,
            projection_pairs_rotation=projection_pairs_rotation,
            frame_ID1=frame_ID1,
            frame_ID2=frame_ID2,
        )
