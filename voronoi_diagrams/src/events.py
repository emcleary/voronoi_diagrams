from __future__ import annotations

import numpy as np

from voronoi_diagrams.src.math_voronoi import circle_parameters
from voronoi_diagrams.src.point import Point2D

from typing import Self, TYPE_CHECKING
if TYPE_CHECKING:
    from .trees.tree_vd import LeafVD


class Event:
    """
    Event base class for a sweepline Voronoi diagram algorithm

    Event classes are used in Fortune's sweepline algorithm for Voronoi
    diagrams for queuing input sites and vertices of the diagram. 
    This base class is specifically for sites.

    Attributes:
        _point (Point2D): A point used for sorting events
    """
    def __init__(self, point: Point2D):
        """
        Initializes the Event class

        Args:
            point (Point2D): A 2D point with a y-coordinate representing the sweepline
        """
        self._point = point

    def __lt__(self, other: Self) -> bool:
        """
        Compares this event point with that of another event.
        The sweepline is taken as a horizontal line, increasing in value.

        Args:
            other (Event): The other event used for comparison

        Return:
            bool: True if this event point is less than the other, false otherwise
        """
        if self._point.y == other._point.y:
            return self._point.x < other._point.x
        return self._point.y < other._point.y


class CircleEvent(Event):
    """
    An event class for vertices of the Voronoi diagram

    Attributes:
        _center: Center of a circle, i.e. the vertex added to the Voronoi diagram
        _radius: Radius of the circle
        _active (bool): True if the event is active, false if overridden by another event
        _node (LeafVD): Node of the Voronoi tree corresponding to this event
    """

    def __init__(self, center: Point2D, radius: float):
        """
        Initializes the class with parameters of a circle

        Args:
            center (Point2D): The center of the circle
            radius (float): The radius of the circle
        """
        super(CircleEvent, self).__init__(Point2D(center.x, center.y + radius))
        self._center = center
        self._radius = radius
        self._active = True
        self._node: LeafVD | None = None

    @property
    def node(self):
        """
        Get the node of the object

        The assertion is included to ensure the node is set (externally)
        and to help with type hinting.
        """
        assert self._node is not None
        return self._node

    @node.setter
    def node(self, node: LeafVD):
        """
        Set the node of the object

        The node should only be set (externally) once, hence the assertion check.
        """
        assert self._node is None
        self._node = node

    def deactivate(self) -> None:
        """
        Deactivate the event

        Event deactivation occurs when a new site is inserted that is within the circle.
        Deactivation should only happen once, hence the assertion check.
        """
        assert self._active
        self._active = False

    def is_active(self) -> bool:
        """
        Check if the event is active
        """
        return self._active

    def contains(self, point: Point2D) -> bool:
        """
        Checks if a point is in the circle

        Args:
            point (Point2D): input point

        Return:
            bool: True if the point is inside the circle or on its circumference, false otherwise
        """
        d = point.distance(self._center)
        return np.less_equal(d, self._radius)


def make_circle_event(p: Point2D, q: Point2D, r: Point2D) -> CircleEvent | None:
    """
    Creates a circle event from 3 points

    Args:
        p (Point2D): input point
        q (Point2D): input point
        r (Point2D): input point

    Return:
        CircleEvent or None: Returns None if the points are colinear, otherwise creates and returns the event
    """
    parameters = circle_parameters(p, q, r)
    if parameters is None:
        return None
    center, radius = parameters
    return CircleEvent(center, radius)
