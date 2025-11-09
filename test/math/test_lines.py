import numpy as np
import pytest

from voronoi_diagrams.src.math_voronoi import line_parameters, perpendicular_line_parameters
from voronoi_diagrams.src.point import Point2D

from typing import Tuple


@pytest.fixture(params=[
    (Point2D(1, 2), Point2D(2, 3)),
    (Point2D(1, 2), Point2D(3, 4)),
    (Point2D(1, 1), Point2D(1, 2)), # vertical
    (Point2D(1, 1), Point2D(2, 1)), # horizontal
])
def points(request):
    return request.param

def test_line_parameters(points: Tuple[Point2D, Point2D]) -> None:
    p, q = points
    a, b, c = line_parameters(p, q)
    assert np.isclose(a*p.x + b*p.y, c)
    assert np.isclose(a*q.x + b*q.y, c)

def test_perpendicular_line_parameters(points: Tuple[Point2D, Point2D]) -> None:
    p, q = points
    ap, bp, _ = perpendicular_line_parameters(p, q)
    a, b, c = line_parameters(p, q)
    assert a == bp
    assert -b == ap
    x = (p.x + q.x) / 2
    y = (p.y + q.y) / 2
    assert np.isclose(a*x + b*y, c)


if __name__ == '__main__':
    p = Point2D(1, 2)
    q = Point2D(2, 3)
    a, b, c = line_parameters(p, q)
    ap, bp, cp = perpendicular_line_parameters(p, q)
    print(a, b, c)
    print(ap, bp, cp)
    assert a == bp
    assert -b == ap
