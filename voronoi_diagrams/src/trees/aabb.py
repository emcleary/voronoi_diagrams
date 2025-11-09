import numpy as np

from voronoi_diagrams.src.point import Point

from typing import Self


# works for anything positive and NONZERO
SKIN = 1e-12

# Axis Aligned Bounding Box
class AABB:
    """
    The axis-aligned bounding box (AABB) class used for BVH trees

    Attributes:
        _pmin [Point]: Point of all minimum coordinates of the bounding box
        _pmax [Point]: Point of all maximum coordinates of the bounding box
        _surface_area [float]: Surface area of the bounding box
    """

    def __init__(self, *points: Point, ndim=None):
        """
        Creates the AABB object

        Args:
            points [Point]: Initial points contained in the bounding box
            ndim [int, optional]: Dimensions of a point
        """
        if ndim is None:
            assert len(points) > 0
            ndim = points[0].dimension
        elif points:
            assert ndim == points[0].dimension
        assert ndim > 0
        self._pmin = Point(*[-np.inf] * ndim)
        self._pmax = Point(*[np.inf] * ndim)
        self._surface_area = 0.0
        if points:
            self._set(*points)

    def _set(self, *points: Point) -> None:
        """
        Set the bounding box dimensions

        Args:
            points [Point]: A set of points contained in the bounding box
        """
        self._pmin._x[:] = np.inf
        self._pmax._x[:] = -np.inf
        self._update(*points)

    def _update(self, *points: Point) -> None:
        """
        Update the bounding box dimensions

        Args:
            points [Point]: A set of points added to the bounding box
        """
        for point in points:
            for i, xi in enumerate(point._x):
                self._pmin._x[i] = min(self._pmin._x[i], xi)
                self._pmax._x[i] = max(self._pmax._x[i], xi)
        self._update_surface_area()

    def union(self, box: Self) -> None:
        """
        Merge this object's bounding box with another

        Args:
            box [AABB]: Another bounding box
        """
        self._update(box._pmin, box._pmax)

    def intersect(self, box: Self) -> bool:
        """
        Check if this object's box intersects with another

        Args:
            box [AABB]: Another bounding box
        """
        return self._pmin <= box._pmax and box._pmin <= self._pmax

    def contains(self, point: Point) -> bool:
        """
        Check if a point is contained within or on the surface of
        this object's box

        Args:
            point [Point]: The test point
        """
        return self._pmin <= point and point <= self._pmax

    @property
    def surface_area(self) -> float:
        """
        Get surface area

        Return:
            float: The surface area of the box
        """
        return self._surface_area

    def _update_surface_area(self) -> None:
        """
        Updates the surface area of the box
        """
        area = 0
        n = len(self._pmin._x)
        for i in range(n):
            a = 1
            for j in range(n):
                if i == j: continue
                dx = self._pmax._x[j] - self._pmin._x[j]
                a *= dx + 2*SKIN
            assert np.greater_equal(a, 0)
            area += a
        self._surface_area = 2 * area

    def proposed_surface_area(self, point: Point) -> float:
        """
        Calculates a surface area if a new point is added

        Args:
            point [Point]: The test point
        """

        # NB: faster to calculate the area than to check first
        # if the box contains the test point, possibly skipping
        # the calculation
        area = 0
        n = len(self._pmin._x)
        for i in range(n):
            a = 1
            for j in range(n):
                if i == j: continue
                xmin = min(self._pmin._x[j], point._x[j])
                xmax = max(self._pmax._x[j], point._x[j])
                dx = xmax - xmin
                a *= dx + 2*SKIN
            area += a
        return 2 * area

    def __lt__(self, other: Self) -> bool:
        """
        Compare the minimum point of this object's box to another

        Args:
            other [AABB]: The other box

        Return:
            bool: True if this object's minimum point is less than that of the other, 
                false otherwise
        """
        if self._pmin == other._pmin:
            return self._pmax < other._pmax
        return self._pmin < other._pmin

    def __str__(self) -> str:
        return f'[{self._pmin} x {self._pmax}]'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}[{self._pmin} x {self._pmax}]'
