from abc import ABC, abstractmethod
import numpy as np

from cardillo.utility.parametrize import parametrize


class CrossSection(ABC):
    """Abstract class definition for rod cross-sections."""

    @abstractmethod
    def area(self, xi):
        """Area of the cross-section."""
        ...

    @abstractmethod
    def first_moment(self, xi):
        """Vector containing the first moments of area."""
        ...

    @abstractmethod
    def second_moment(self, xi):
        """Matrix containing the second moments of area."""
        ...


class ExportableCrossSection(CrossSection):
    @abstractmethod
    def create_mesh(self, xis): ...


class UserDefinedCrossSection(CrossSection):
    def __init__(self, area, first_moment, second_moment):
        """User defined cross-section.

        Parameters
        ----------
        area : float or callable(xi) -> float
            Area of the cross-section.
        first_moment : np.ndarray (3,) or callable(xi) -> np.ndarray (3,)
            Vector containing the first moments of area.
        second_moment : np.ndarray (3, 3) or callable(xi) -> np.ndarray (3, 3)
            Matrix containing the second moments of area.
        """
        self._area = parametrize(area)
        self._first_moment = parametrize(first_moment)
        self._second_moment = parametrize(second_moment)

    def area(self, xi):
        return self._area(xi)

    def first_moment(self, xi):
        return self._first_moment(xi)

    def second_moment(self, xi):
        return self._second_moment(xi)


class CircularCrossSection(ExportableCrossSection):
    def __init__(self, radius, *, export_resolution=16):
        """Circular cross-section.

        Parameters
        ----------
        radius : float or callable(xi) -> float
            Radius of the cross-section (at xi).
        """
        self.radius = parametrize(radius)
        self._second_moment = np.pi / 4 * np.diag([2, 1, 1])

        self.export_resolution = export_resolution

    def area(self, xi):
        r = self.radius(xi)
        return np.pi * r**2

    def first_moment(self, xi):
        # see https://en.wikipedia.org/wiki/First_moment_of_area
        xi = np.asarray(xi)
        return np.zeros(xi.shape + (3,))

    def second_moment(self, xi):
        # https://en.wikipedia.org/wiki/List_of_second_moments_of_area
        r = np.asarray(self.radius(xi))
        return (r**4)[..., None, None] * self._second_moment

    def create_mesh(self, xis):
        ncirc = self.export_resolution

        verts = []
        indices = []
        joints = []
        weights = []
        for j, xi in enumerate(xis):
            radius = self.radius(xi)
            for i in range(ncirc):
                a = 2 * np.pi * i / ncirc
                y = radius * np.cos(a)
                z = radius * np.sin(a)

                # cardillo to glTF: y <- z and z <- -y
                verts.append([0.0, z, -y])
                joints.append([j, 0, 0, 0])
                weights.append([1.0, 0.0, 0.0, 0.0])

        # indices (grid)
        for j in range(len(xis) - 1):
            for i in range(ncirc):
                i0 = j * ncirc + i
                i1 = j * ncirc + (i + 1) % ncirc
                i2 = (j + 1) * ncirc + i
                i3 = (j + 1) * ncirc + (i + 1) % ncirc

                indices.extend([i0, i1, i2])
                indices.extend([i1, i3, i2])

        # close surfaces
        last = (len(xis) - 1) * ncirc
        for i in range(ncirc - 2):
            indices.extend([0, i + 2, i + 1])
            indices.extend([last, last + i + 2, last + i + 1])

        return (
            np.array(verts, np.float32),
            np.array(indices, np.uint32),
            np.array(joints, np.uint16),
            np.array(weights, np.float32),
        )


class RectangularCrossSection(ExportableCrossSection):
    def __init__(self, width, height):
        """Rectangular cross-section.

        Parameters:
        -----
        width : float or callable(xi) -> float
            Cross-section dimension in e_y^B-direction (at xi).
        height : float or callable(xi) -> float
            Cross-section dimension in e_z^B-direction (at xi).
        """
        self.width = parametrize(width)
        self.height = parametrize(height)

    def area(self, xi):
        return self.width(xi) * self.height(xi)

    def first_moment(self, xi):
        # see https://en.wikipedia.org/wiki/First_moment_of_area
        xi = np.asarray(xi)
        return np.zeros(xi.shape + (3,))

    def second_moment(self, xi):
        # https://en.wikipedia.org/wiki/List_of_second_moments_of_area
        width = self.width(xi)
        height = self.height(xi)

        Iyy = width * height**3 / 12.0
        Izz = width**3 * height / 12.0
        Ixx = Iyy + Izz

        second_moment = np.zeros(np.asarray(width).shape + (3, 3))
        second_moment[..., 0, 0] = Ixx
        second_moment[..., 1, 1] = Iyy
        second_moment[..., 2, 2] = Izz
        return second_moment

    def create_mesh(self, xis):
        verts = []
        indices = []
        joints = []
        weights = []
        for j, xi in enumerate(xis):
            y_ = self.width(xi) / 2
            z_ = self.height(xi) / 2
            for i in range(4):
                y = y_ * (1 if i % 4 in (1, 2) else -1)
                z = z_ * (-1 if i % 4 < 2 else 1)

                # cardillo to glTF: y <- z and z <- -y
                verts.append([0.0, z, -y])
                joints.append([j, 0, 0, 0])
                weights.append([1.0, 0.0, 0.0, 0.0])

        # indices (grid)
        for j in range(len(xis) - 1):
            for i in range(4):
                i0 = j * 4 + i
                i1 = j * 4 + (i + 1) % 4
                i2 = (j + 1) * 4 + i
                i3 = (j + 1) * 4 + (i + 1) % 4

                indices.extend([i0, i1, i2])
                indices.extend([i1, i3, i2])

        # close surfaces
        last = (len(xis) - 1) * 4
        for i in range(2):
            indices.extend([0, i + 2, i + 1])
            # TODO: why is this swapepd?
            indices.extend([last, last + i + 1, last + i + 2])

        return (
            np.array(verts, np.float32),
            np.array(indices, np.uint32),
            np.array(joints, np.uint16),
            np.array(weights, np.float32),
        )


class CrossSectionInertias:
    def __init__(
        self, density=None, cross_section=None, A_rho0=1.0, B_I_rho0=np.eye(3)
    ):
        """Inertial properties of cross-sections. Centerline must coincide with line of centroids.

        Parameters:
        -----
        density : float or callable(xi) -> float
            Mass per unit reference volume of the rod.
        cross_section : CrossSection
            Cross-section object, which provides cross-section area and second moment of area.
        A_rho0 : float or callable(xi) -> float
            Cross-section mass density, i.e., mass per unit reference length of rod.
        B_I_rho0 : np.ndarray(3, 3) or  callable(xi) -> np.ndarray(3, 3)
            Cross-section inertia tensor represented in the cross-section-fixed B-Basis.

        """
        if density is None or cross_section is None:
            self.A_rho0 = parametrize(A_rho0)
            self.B_I_rho0 = parametrize(B_I_rho0)
        else:
            _density = parametrize(density)
            self.A_rho0 = lambda xi: _density(xi) * cross_section.area(xi)
            self.B_I_rho0 = lambda xi: _density(xi)[
                ..., None, None
            ] * cross_section.second_moment(xi)
