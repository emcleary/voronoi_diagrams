import numpy as np
import pytest
from typing import Tuple

from voronoi_diagrams.src.math_voronoi import parabola_y
from voronoi_diagrams.src.point import Point2D


@pytest.fixture(params=[-3, -1, 0, 1, 3])
def x(request):
    return request.param

@pytest.fixture(params=[
    (Point2D(0, 4), 5),
    (Point2D(0, 6), 5),
    (Point2D(0, 5), 5), # focus on directrix -> y=inf
])
def parabola(request):
    return request.param

def test_parabola(parabola: Tuple[Point2D, float], x: float) -> None:
    focus, directrix = parabola
    y = parabola_y(focus, directrix, x)
    p = Point2D(x, y)
    r = Point2D(x, directrix)
    assert np.isclose(p.distance(focus), p.distance(r))

if __name__ == '__main__':
    focus = Point2D(0, 5)
    directrix = 5.0
    x = 1
    y = parabola_y(focus, directrix, x)
    p = Point2D(x, y)
    r = Point2D(x, directrix)
    print(x, y)
    print(p)
    print(focus)
    print(r)
    assert np.isclose(p.distance(focus), p.distance(r))
