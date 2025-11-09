from voronoi_diagrams.src.math_voronoi import is_left, is_right, is_on_line
from voronoi_diagrams.src.point import Point2D


def test_is_left() -> None:
    p = Point2D(0, 0)
    q = Point2D(0, 1)
    rl = Point2D(-1, 1)
    rr = Point2D(1, 1)
    assert is_left(p, q, rl)
    assert not is_left(p, q, rr)


def test_is_right() -> None:
    p = Point2D(0, 0)
    q = Point2D(0, 1)
    rl = Point2D(-1, 1)
    rr = Point2D(1, 1)
    assert not is_right(p, q, rl)
    assert is_right(p, q, rr)
    

def test_is_on_line() -> None:
    px = Point2D(1, 0)
    qx = Point2D(2, 0)
    rx = Point2D(3, 0)

    py = Point2D(0, 1)
    qy = Point2D(0, 2)
    ry = Point2D(0, 3)
    
    assert is_on_line(px, qx, rx)
    assert is_on_line(py, qy, ry)
    assert not is_on_line(px, qx, ry)
    assert not is_on_line(px, qy, rx)
    assert not is_on_line(py, qx, rx)
    assert not is_on_line(py, qy, rx)
    assert not is_on_line(py, qx, ry)
    assert not is_on_line(px, qy, ry)

    eps = 1e-7
    rl = Point2D(-eps, 3)
    rr = Point2D(eps, 3)
    assert not is_on_line(py, qy, rl)
    assert not is_on_line(py, qy, rr)

    eps = 1e-8
    rl = Point2D(-eps, 3)
    rr = Point2D(eps, 3)
    assert is_on_line(py, qy, rl)
    assert is_on_line(py, qy, rr)
