from dataclasses import dataclass, field
import pytest

from voronoi_diagrams.src.point import Point2D
from voronoi_diagrams.src.trees.node import Internal
from voronoi_diagrams.src.trees.tree_vd import TreeVD, LeafVD, InternalVD

from typing import List


@dataclass
class Data:
    points: list
    to_delete: list
    sweepline_all: float
    sweepline_delete: float
    tree: TreeVD = field(init=False)

    def __post_init__(self):
        self.tree = TreeVD()
        self.points.sort(key=lambda p: [p[1], p[0]])
        for point in self.points:
            self.tree.insert(Point2D(*point))

    def delete(self):
        self.to_delete.sort()

        node = self.tree._root
        while isinstance(node, Internal):
            node = node.left
        assert isinstance(node, LeafVD)

        i = 0
        nodes: List[LeafVD] = []
        for j in self.to_delete:
            while i < j:
                node = self.tree.get_successor(node)
                assert node is not None
                i += 1
            nodes.append(node)

        for node in nodes:
            self.tree.delete_node(node)
        
data0 = Data(
    [
        (0, 0),
        (-1, 1),
        (1, 1),
        (0, 1.1)
    ],
    [2, 4], # parabolas with focus at x = 0
    1.12, # before event removing inner parabolas
    1.3, # after event removing inner parabolas
)
    
data1 = Data(
    [
        (-10, 1),
        (-9, 9),
        (3, 6),
        (-2, 9),
    ],
    [3], # parabolas with focus at x = 0
    9.1, # before event removing inner parabola
    10, # after event removing inner parabola
)
    

@pytest.fixture(
    params=[
        data1,
    ],
    scope='module',    
)
def data(request) -> Data:
    return request.param

@pytest.fixture(scope='module')
def data_deleted(data: Data) -> Data:
    data.delete()
    return data

def assert_correct_heights(tree: TreeVD) -> None:
    for node in tree.get_internals():
        assert node.height == 1 + max(node.left.height, node.right.height)

def assert_correct_internals(tree: TreeVD, sweepline: float) -> None:
    for node in tree.get_internals():
        x, y = node.calculate_breakpoint(sweepline)
        if isinstance(node.left, InternalVD):
            xl, _ = node.left.calculate_breakpoint(sweepline)
            assert xl < x
        if isinstance(node.right, InternalVD):
            xr, _ = node.right.calculate_breakpoint(sweepline)
            assert xr > x

###################
# BEFORE DELETION #
###################
            
def test_height(data: Data) -> None:
    assert_correct_heights(data.tree)

def test_internal(data: Data) -> None:
    tree = data.tree
    sweepline = data.sweepline_all
    assert_correct_internals(tree, sweepline)

##################
# AFTER DELETION #
##################

def test_height_deleted(data_deleted: Data) -> None:
    assert_correct_heights(data_deleted.tree)

def test_internal_deleted(data_deleted: Data) -> None:
    tree = data_deleted.tree
    sweepline = data_deleted.sweepline_delete
    assert_correct_internals(tree, sweepline)
