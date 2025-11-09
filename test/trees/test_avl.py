import pytest

from voronoi_diagrams.src.trees.tree_avl import ScalarTreeAVL, LeafAVL, InternalAVL

from typing import Sequence

@pytest.fixture(params=[
    # rotations
    (1, 2, 3, 4, 5), # LL
    (5, 4, 3, 2, 1), # RR
    (8, 7, 5, 6), # LR
    (5, 8, 7, 6), # RL
    # repeats
    (1, 1, 1, 1, 1),
    (5, 4, 4, 8, 9, 1, 10),
])
def inputs(request) -> Sequence[int]:
    return request.param

@pytest.fixture
def tree(inputs: Sequence[int]) -> ScalarTreeAVL[int]:
    tree = ScalarTreeAVL[int]()
    for value in inputs:
        tree.insert(value)
    return tree

def test_balance(tree: ScalarTreeAVL[int]) -> None:
    assert tree._root is not None
    assert abs(tree._root.imbalance) < 2

def test_all_leaves(tree: ScalarTreeAVL[int], inputs: Sequence[int]) -> None:
    n = 0
    for node in tree.get_leaves():
        assert isinstance(node, LeafAVL)
        n += 1
    assert n == len(inputs)

def test_all_internals(tree: ScalarTreeAVL[int], inputs: Sequence[int]) -> None:
    n = 0
    for node in tree.get_internals():
        v0, v1 = node._internal
        assert v0 <= v1
        pred = tree.get_predecessor(node)
        succ = tree.get_successor(node)
        assert pred is not None
        assert succ is not None
        assert pred._value == v0
        assert succ._value == v1
        n += 1
    assert n == len(inputs) - 1

def test_predecessor(tree: ScalarTreeAVL[int]) -> None:
    node = tree._root
    while isinstance(node, InternalAVL):
        node = node._right
    assert isinstance(node, LeafAVL)
    current = node.value
    node = tree.get_predecessor(node)
    while node is not None:
        assert node.value <= current
        node = tree.get_predecessor(node)

def test_successor(tree: ScalarTreeAVL[int]) -> None:
    node = tree._root
    while isinstance(node, InternalAVL):
        node = node._left
    assert isinstance(node, LeafAVL)
    current = node.value
    node = tree.get_successor(node)
    while node is not None:
        assert node.value >= current
        node = tree.get_successor(node)

def test_height(tree):
    for node in tree.get_internals():
        assert node.height == 1 + max(node._left.height, node._right.height)


if __name__ == '__main__':
    t = ScalarTreeAVL[int]()
    values = [
        1, 2, 3, 4, 5
    ]
    values = [
        5, 4, 3, 2, 1
    ]

    values = [
        8, 7, 5, 6
    ]

    values = [
        5, 8, 7, 6
    ]

    for value in values:
        t.insert(value)

    t.plot()
    test_balance(t)
    test_all_leaves(t, values)
    test_all_internals(t, values)
    test_predecessor(t)
    test_successor(t)
