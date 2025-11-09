import numpy as np
import pytest

from voronoi_diagrams.src.math_voronoi import get_parabola_intersection
from voronoi_diagrams.src.point import Point2D

from typing import Tuple


@pytest.fixture(params=[
    (Point2D(0, 1), Point2D(1, 2)),
    (Point2D(0, 1), Point2D(1, 2)),
    (Point2D(0, 1), Point2D(1, 2)),
])
def parabolas(request):
    return request.param

@pytest.fixture(params=[
    1e-4, 1e-2, 1e-1, 1, 10, 100
])
def offset(request):
    return request.param

def assert_equal_distance(f0: Point2D, f1: Point2D, 
                          on_dir: Point2D, at_int: Point2D) -> None:
    d_f0 = f0.distance(at_int)
    d_f1 = f1.distance(at_int)
    d_dir = on_dir.distance(at_int)
    assert np.isclose(d_f0, d_f1)
    assert np.isclose(d_f0, d_dir)
    assert np.isclose(d_f1, d_dir)

def assert_correct_intersection(f0: Point2D, f1: Point2D, 
                                p: Point2D) -> None:
    xmin = min(f0.x, f1.x)
    xmax = max(f0.x, f1.x)
    if f0.y == f1.y:
        assert p.y < f0.y
        assert xmin < p.x and p.x < xmax
    elif f0.y < f1.y:
        assert np.less_equal(p.x, f1.x)
    else:
        assert np.greater_equal(p.x, f0.x)

def test_intersection(parabolas: Tuple[Point2D, Point2D], offset: float) -> None:
    assert offset > 0
    focus0, focus1 = parabolas
    directrix = offset + max(focus0.y, focus1.y)

    x01, y01 = get_parabola_intersection(focus0, focus1, directrix)
    point_on_directrix = Point2D(x01, directrix)
    point_at_intersection = Point2D(x01, y01)
    assert_equal_distance(focus0, focus1, point_on_directrix, point_at_intersection)
    assert_correct_intersection(focus0, focus1, point_at_intersection)

    x10, y10 = get_parabola_intersection(focus1, focus0, directrix)
    point_on_directrix = Point2D(x10, directrix)
    point_at_intersection = Point2D(x10, y10)
    assert_equal_distance(focus0, focus1, point_on_directrix, point_at_intersection)
    assert_correct_intersection(focus1, focus0, point_at_intersection)

def test_directrix_at_point(parabolas: Tuple[Point2D, Point2D]) -> None:
    focus0, focus1 = parabolas
    directrix = max(focus0.y, focus1.y)
    target = focus0.x if focus0.y > focus1.y else focus1.x
    x01, y01 = get_parabola_intersection(focus0, focus1, directrix)
    assert np.isclose(x01, target)
    assert y01 < directrix
    x10, y10 = get_parabola_intersection(focus1, focus0, directrix)
    assert np.isclose(x10, target)
    assert y10 < directrix

def test_no_intersection() -> None:
    directrix = 1
    focus0 = Point2D(1, directrix)
    focus1 = Point2D(2, directrix)
    x01, y01 = get_parabola_intersection(focus0, focus1, directrix)
    assert x01 == np.inf
    assert y01 == np.inf
    x10, y10 = get_parabola_intersection(focus1, focus0, directrix)
    assert x10 == np.inf
    assert y10 == np.inf


if __name__ == '__main__':
    f0 = Point2D(0, 1)
    f1 = Point2D(1, 2)
    offsets = [1e-4, 1e-2, 1, 2, 4]
    dir = max(f0.y, f1.y)
    for o in offsets:
        test_intersection((f0, f1), o)