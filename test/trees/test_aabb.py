from voronoi_diagrams.src.point import Point
from voronoi_diagrams.src.trees.aabb import AABB


def test_union():
    p0 = Point(0, 0)
    p1 = Point(1, 1)
    box01 = AABB(p0, p1)

    p2 = Point(2, 2)
    p3 = Point(3, 3)
    box23 = AABB(p2, p3)

    p4 = Point(1, 2)
    p5 = Point(2, 1)
    assert not box01.contains(p4)
    assert not box01.contains(p5)
    assert not box23.contains(p4)
    assert not box23.contains(p5)

    box01.union(box23)
    assert box01.contains(p4)
    assert box01.contains(p5)
    assert not box23.contains(p4)
    assert not box23.contains(p5)


def test_intersect():
    p0 = Point(0, 0)
    p1 = Point(2, 2)
    box01 = AABB(p0, p1)

    p2 = Point(2, 2)
    p3 = Point(4, 4)
    box23 = AABB(p2, p3)
    
    p4 = Point(1, 3)
    p5 = Point(3, 1)
    box45 = AABB(p4, p5)

    eps = 1e-8
    p6 = Point(*(p1._x + eps))
    p7 = Point(3, 3)
    box67 = AABB(p6, p7)

    assert box01.intersect(box23)
    assert box23.intersect(box01)
    assert box01.intersect(box45)
    assert box45.intersect(box01)
    assert box23.intersect(box45)
    assert box45.intersect(box23)
    assert not box01.intersect(box67)
    assert not box67.intersect(box01)
    

def test_contains():
    p0 = Point(0, 0)
    p1 = Point(2, 2)
    box01 = AABB(p0, p1)

    assert box01.contains(p0)
    assert box01.contains(p1)

    p2 = Point(0, 1)
    p3 = Point(2, 1)
    assert box01.contains(p2)
    assert box01.contains(p3)

    p4 = Point(1, 0)
    p5 = Point(1, 2)
    assert box01.contains(p4)
    assert box01.contains(p5)

    p6 = Point(1, 1)
    p7 = Point(3, 3)
    p8 = Point(-1, -1)
    assert box01.contains(p6)
    assert not box01.contains(p7)
    assert not box01.contains(p8)


if __name__ == '__main__':
    test_intersect()
    test_contains()
