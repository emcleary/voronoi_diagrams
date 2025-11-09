import numpy as np
import pytest

from voronoi_diagrams.src.math_voronoi import circle_parameters, is_on_line
from voronoi_diagrams.src.point import Point2D

from typing import Tuple


@pytest.fixture(params=[
    (0, 90, 180),
    (1, 2, 3),
])
def angles(request) -> float:
    return request.param

def calculate_point(angle: float) -> Point2D:
    angle *= np.pi / 180
    x = np.cos(angle)
    y = np.sin(angle)
    return Point2D(x, y)

@pytest.fixture
def points(angles: Tuple[float, float, float]) -> Tuple[Point2D, Point2D, Point2D]:
    ratio = np.pi / 180
    p = calculate_point(angles[0])
    q = calculate_point(angles[1])
    r = calculate_point(angles[2])
    return p, q, r

def test_circle(points: Tuple[Point2D, Point2D, Point2D]) -> None:
    parameters = circle_parameters(*points)
    assert parameters is not None
    center, radius = parameters
    assert np.isclose(radius, 1)
    assert np.isclose(center.x, 0)
    assert np.isclose(center.y, 0)

def test_colinear() -> None:
    p = Point2D(0, 1)
    q = Point2D(1, 2)
    r = Point2D(2, 3)
    assert is_on_line(p, q, r)
    result = circle_parameters(p, q, r)
    assert result is None
