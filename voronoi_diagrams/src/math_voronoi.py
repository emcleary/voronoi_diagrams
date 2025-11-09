import numpy as np

from voronoi_diagrams.src.point import Point2D

from typing import Tuple


def line_parameters(p: Point2D, q: Point2D) -> Tuple[float, float, float]:
    """
    Calculate parameters for a line

    Calculates line parameters (a, b, c) that satisfy the equation a*x + b*y = c.

    Args:
        p (Point2D): A point in 2-dimensions
        q (Point2D): A point in 2-dimensions

    Return:
        Tuple[float, float, float]: Parameters a, b, c
    """
    dx = q.x - p.x
    dy = q.y - p.y
    c = -dy*p.x + dx*p.y
    return -dy, dx, c


def perpendicular_line_parameters(p: Point2D, q: Point2D) -> Tuple[float, float, float]:
    """
    Calculates parameters for a perpendicular line

    Calculates line parameters (a, b, c) that satisfy the equation a*x + b*y + c
    and interssects the midpoint between the 2 input points.

    Args:
        p (Point2D): A point in 2-dimensions
        q (Point2D): A point in 2-dimensions

    Return:
        Tuple[float, float, float]: Parameters a, b, c
    """
    dx = q.x - p.x
    dy = q.y - p.y
    xm = (p.x + q.x) / 2
    ym = (p.y + q.y) / 2
    c = -dx*xm - dy*ym
    return -dx, -dy, c


def circle_parameters(p: Point2D, q: Point2D, r: Point2D) -> Tuple[Point2D, float] | None:
    """
    Calculates the center point and radius of a circle intersecting 3 points

    Args:
        p (Point2D): A point in 2-dimensions
        q (Point2D): A point in 2-dimensions
        r (Point2D): A point in 2-dimensions

    Return:
        Tuple[Point2D, float] or None: Returns None if the points are colinear,
            otherwise returns the center point and the radius of the circle.
    """
    a0, b0, c0 = perpendicular_line_parameters(p, q)
    a1, b1, c1 = perpendicular_line_parameters(p, r)
    det = a0 * b1 - a1 * b0
    if np.isclose(det, 0): # colinear points
        return None
    A = np.array([[a0, b0], [a1, b1]])
    Y = np.array([c0, c1])
    x, y = np.linalg.solve(A, Y)
    dx = p.x - x
    dy = p.y - y
    radius = np.sqrt(dx*dx + dy*dy)
    return Point2D(x, y), radius
    

def parabola_y(focus: Point2D, directrix: float, x: float) -> float:
    """
    Calculate the y-coordinate of a parabola

    The focus and directrix define the parabola. The directrix is assumed
    to be a horizontal line. Given an x-coordinate, this function calculates
    the corresponding y-coordinate on the parabola.

    Args:
        focus (Point2D): The focus of a parabola
        directrix (float): The directrix of the parabola
        x (float): The x-coordinate of the point on the parabola

    Return:
        y (float): The y-coordinate of the point on the parabola
    """
    if np.isclose(focus.y, directrix):
        return np.inf
    dx = x - focus.x
    dy = directrix - focus.y
    b = directrix + focus.y
    return (b - dx*dx/dy) / 2


def get_parabola_intersection(f0: Point2D, f1: Point2D,
                              directrix: float) -> Tuple[float, float]:
    """
    Calculate a parabola intersection

    Unless both foci have the same y-value (hence distance from the directrix),
    there will be 2 intersections between parabolas. This method only returns
    1 intersection, corresponding to the order of foci in the arguments and their
    relative positions. This is done for efficiency while calculating breakpoints
    in the Voronoi diagram tree. The directrix is assumed to be a horizontal line
    and is at or above the y-coordinate of the higher focus point. Coordinates of
    +infinity are returned if there is no intersection.

    Args:
        f0 (Point2D): A focus point
        f1 (Point2D): A focus point
        directrix (float): A directrix

    Return:
        Tuple[float, float]: x, y coordinates of the desired intersection
    """

    if np.isclose(f0.y, directrix) and np.isclose(f1.y, directrix):
        # both parabolas are singularities, hence no intersection
        return np.inf, np.inf
    
    assert np.less_equal(f0.y, directrix)
    assert np.less_equal(f1.y, directrix)
    
    Y0d = f0.y - directrix
    Y1d = f1.y - directrix

    A = Y1d - Y0d
    B = 2 * (f1.x*Y0d - f0.x*Y1d)
    C = f0.x*f0.x*Y1d - f1.x*f1.x*Y0d - A*Y0d*Y1d

    if np.isclose(A, 0):
        # Both foci are the same distance from the directrix.
        # Hence only 1 intersection.
        # NB: B is guaranteed to be nonzero since the y-coordinates of
        # foci are less than the directrix and both foci are at
        # different x-coordinates
        x = -C / B
        y = parabola_y(f0, directrix, x)
        return x, y

    discriminant = B*B - 4*A*C
    if discriminant < 0:
        assert np.isclose(B*B, 4*A*C)
        discriminant = 0

    temp = np.sqrt(discriminant)
    d_minus = (-B - temp) / 2 / A
    d_plus = (-B + temp) / 2 / A
    if f0.y > f1.y:
        x = max(d_minus, d_plus)
    else:
        x = min(d_minus, d_plus)

    if np.isclose(f0.y, directrix):
        y = parabola_y(f1, directrix, x)
    else:
        y = parabola_y(f0, directrix, x)

    return x, y


def det(p: Point2D, q: Point2D, r: Point2D) -> float:
    """
    Determinant of 3 points

    Args:
        p (Point2D): A point in 2-dimensions
        q (Point2D): A point in 2-dimensions
        r (Point2D): A point in 2-dimensions                

    Return:
        float: The determinant
    """
    return (p.x - r.x) * (q.y - r.y) - (p.y - r.y) * (q.x - r.x)


"""
Check if point r is to the right of line pq
"""
def is_right(p: Point2D, q: Point2D, r: Point2D) -> bool:
    """
    Check if point r is to the right of line pq

    Args:
        p (Point2D): A point in 2-dimensions
        q (Point2D): A point in 2-dimensions
        r (Point2D): A point in 2-dimensions; the test point

    Return:
        bool: True if r is to the right of pq, false otherwise
    """
    d = det(p, q, r)
    return np.less(d, 0) and not np.isclose(d, 0)


def is_left(p: Point2D, q: Point2D, r: Point2D) -> bool:
    """
    Check if point r is to the left of line pq

    Args:
        p (Point2D): A point in 2-dimensions
        q (Point2D): A point in 2-dimensions
        r (Point2D): A point in 2-dimensions; the test point

    Return:
        bool: True if r is to the left of pq, false otherwise
    """
    d = det(p, q, r)
    return np.greater(d, 0) and not np.isclose(d, 0)


def is_on_line(p: Point2D, q: Point2D, r: Point2D) -> bool:
    """
    Check if point r is colinear with the line pq

    Args:
        p (Point2D): A point in 2-dimensions
        q (Point2D): A point in 2-dimensions
        r (Point2D): A point in 2-dimensions; the test point

    Return:
        bool: True if r is colinear with pq, false otherwise
    """
    d = det(p, q, r)
    return bool(np.isclose(d, 0))