import pytest

from voronoi_diagrams.src.point import Point
from voronoi_diagrams.src.trees.tree_bvh import TreeBVH, BalancedTreeBVH, LeafBVH, InternalBVH

from typing import List, Tuple


@pytest.fixture(params=[(-100, -10), (10, 10)], scope='module')
def target(request):
    return Point(*request.param)

@pytest.fixture(params=[
    [(1, 2)] * 10,
    [(1, 2), (3, 4)],
], scope='module')
def data(request) -> List[Tuple[float, float]]:
    return request.param

@pytest.fixture(scope='module')
def points(data: List[Tuple[float, float]]) -> List[Point]:
    points = []
    for d in data:
        points.append(Point(*d))
    return points

@pytest.fixture(scope='module')
def tree(points: List[Point]) -> TreeBVH:
    tree = TreeBVH[Point]()
    for point in points:
        tree.insert(point)
    return tree

def assert_count(tree: TreeBVH[Point] | BalancedTreeBVH[Point]) -> None:
    for node in tree.get_internals():
        assert isinstance(node.left, InternalBVH | LeafBVH)
        assert isinstance(node.right, InternalBVH | LeafBVH)
        assert node.count == 1 + node.left.count + node.right.count

def assert_query_contains(tree: TreeBVH | BalancedTreeBVH, *points: Point) -> None:
    for point in points:
        node = tree.query(point, radius=1e-13)
        assert node is not None
        assert node.value == point

def test_count(tree: TreeBVH | BalancedTreeBVH) -> None:
    assert_count(tree)

def test_query_contains_points(tree: TreeBVH | BalancedTreeBVH, points: List[Point]) -> None:
    assert_query_contains(tree, *points)

def test_query_omitted_points(tree: TreeBVH | BalancedTreeBVH, target: Point) -> None:
    node = tree.query(target, radius=1e-13)
    assert node is None

@pytest.fixture(params=[10, 20, 100])
def size(request) -> int:
    return request.param

# can vary a bit with random points
# so stick to identical points
def test_balanced_tree_height(size: int) -> None:
    tree = BalancedTreeBVH[Point]()
    for _ in range(size):
        tree.insert(Point(1, 2))
    assert_count(tree)
    for node in tree.get_internals():
        assert abs(node.imbalance) <= 2


@pytest.fixture(params=[2, 3, 4, 5])
def dimension(request) -> int:
    return request.param

def test_higher_dimensions(dimension, size) -> None:
    import random

    n = size
    d = dimension
    tree = TreeBVH[Point]()
    points = []
    for _ in range(n):
        values = []
        for _ in range(d):
            values.append(random.randint(0, n))
        points.append(Point(*values))
        tree.insert(points[-1])
    
    assert_count(tree)
    assert_query_contains(tree, *points)



if __name__ == '__main__':
    inputs = [(3, 4), (4, 5)]
    pts = []
    for i in inputs:
        pts.append(Point(*i))
    t = TreeBVH()
    for p in pts:
        t.insert(p)
    test_count(t)
    test_query_contains_points(t, pts)
    omitted = [Point(-100, -10), Point(10, 10)]
    for p in omitted:
        test_query_omitted_points(t, p)
    for n in [10, 20, 100]:
        test_balanced_tree_height(n)
    test_balanced_tree_height(10)
