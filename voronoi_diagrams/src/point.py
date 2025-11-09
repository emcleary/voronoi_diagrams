import numpy as np

from typing import Self


class Point:
    """
    An n-dimensional point

    Attributes:
        _x (np.ndarray[float]): Coordinates of an n-dimensional point
    """

    def __init__(self, *x: float):
        """
        Creates an n-dimensional point

        Args:
            *x (float): N coordinates for the point
        """
        self._x = np.copy(x)
        assert(self._x.ndim == 1)

    def distance(self, other: Self) -> np.floating:
        """
        Calculates Euclidean distance between this point and another

        Args:
            other (Point): The other point

        Return:
            float: Distance between 2 points
        """
        return np.linalg.norm(self._x - other._x)

    @property
    def dimension(self):
        """
        Gets the dimension of the point

        Return:
            int: The dimension of the point
        """
        return len(self._x)

    # NB: from testing, iterating like this is faster than
    # numpy equavlents for array sizes up to ~30, plus it
    # simplifies type hinting
    def __eq__(self, other: object) -> bool:
        """
        Checks if this point equals another

        Args:
            other (Point): The other point

        Return:
            bool: True if points are equal, false otherwise
        """
        if not isinstance(other, Point):
            return NotImplemented
        for a, b in zip(self._x, other._x):
            if a != b:
                return False
        return True

    def __lt__(self, other: Self) -> bool:
        """
        Checks if this point is less than another

        Args:
            other (Point): The other point

        Return:
            bool: True if less than the other point, false otherwise
        """
        for a, b in zip(self._x, other._x):
            if a >= b:
                return False
        return True

    def __le__(self, other: Self) -> bool:
        """
        Checks if this point is less than or equals another

        Args:
            other (Point): The other point

        Return:
            bool: True if less than or equals the other point, false otherwise
        """
        for a, b in zip(self._x, other._x):
            if a > b:
                return False
        return True

    def __str__(self) -> str:
        values = ', '.join(map(str, self._x))
        return f'({values})'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self}'


class Point2D(Point):
    """
    An 2-dimensional point

    Attributes:
        _x (np.ndarray[float]): 2-dimensional point
    """

    def __init__(self, x: float, y: float):
        """
        Creates a 2-dimensional point

        Args:
            x (float): The x-coordinate
            y (float): The y-coordinate
        """
        super(Point2D, self).__init__(x, y)

    @property
    def x(self) -> float:
        """
        Get the x-coordinate of the point

        Return:
            float: x-coordinate
        """
        return self._x[0]

    @property
    def y(self) -> float:
        """
        Get the y-coordinate of the point

        Return:
            float: y-coordinate
        """
        return self._x[1]        

