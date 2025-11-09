from voronoi_diagrams.src.protocol import SupportLT
from voronoi_diagrams.src.trees.node import Leaf, Internal
from voronoi_diagrams.src.trees.tree import Tree

from typing import List, cast


class LeafAVL[T](Leaf[T, List[T]]):
    """
    A leaf object used for the AVL trees

    A leaf object with typing specifically used for AVL trees.
    T is the type of objects to be insert and are also used for leaves.
    Internal nodes contain a list of type T, specifically of length 2
    (List rather than Tuple as mutability is necessary).
    """
    
    def __init__(self, value: T):
        """
        Construct a LeafAVL object

        Args:
            value (T): Data stored in the leaf
        """
        super(LeafAVL, self).__init__(value)

    def __str__(self):
        return str(self._value)


class InternalAVL[T](Internal[T, List[T]]):
    """
    An internal object used for AVL trees

    An internal object with typing specifically intended for AVL trees.
    T is the type of objects to be insert and are also used for leaves.
    Internal nodes contain a list of type T, specifically of length 2
    (List rather than Tuple as mutability is necessary).
    """
    def __init__(self, x: T, y: T):
        """
        Construct an InternalAVL object

        Args:
            x (T): Data stored in the predecessor leaf of this node
            y (T): Data stored in the successor leaf of this node
        """
        super(InternalAVL, self).__init__([x, y])
    
    def __str__(self):
        v0, v1 = self._internal
        return str(f'[{v0}, {v1}]')


class TreeAVL[T, L: LeafAVL, I: InternalAVL](Tree[T, L, I]):
    """
    An tree using AVL balancing methods
    
    This class implements some common AVL rebalancing methods.
    Methods for insertion are left to implemented in derived classes.
    """
    def __init__(self):
        """
        Constructor for an AVL tree        
        """
        super(TreeAVL, self).__init__()

    def _rebalance(self, leaf: LeafAVL[T]) -> None:
        """
        Rebalances the tree

        Args:
            leaf (LeafAVL[T]): A node to start the rebalancing
        """
        node = leaf.parent
        while True:
            assert isinstance(node, InternalAVL)
            if node.imbalance == 2:
                if node.left.imbalance == -1:
                    assert isinstance(node.left, InternalAVL)
                    self._rotate_rr(node.left)
                self._rotate_ll(node)
            elif node.imbalance == -2:
                if node.right.imbalance == 1:
                    assert isinstance(node.right, InternalAVL)
                    self._rotate_ll(node.right)
                self._rotate_rr(node)

            node.update_height()
            if node is self._root:
                break
            node = node.parent

    def _rotate_ll(self, node: InternalAVL[T]) -> None:
        """
        Rotates the tree to the right (deepest on "left left" branch)

        Args:
            node (InternalAVL[T]): Node to rotate
        """
        assert isinstance(node.left, Internal)
        left = node.left
        leftright = left.right

        if node is self._root:
            self._root = cast(I, left)
            left.parent = None
        else:
            left.parent = node.parent
            if node.parent.left is node:
                left.parent.left = left
            else:
                assert node.parent.right is node
                left.parent.right = left

        left.right = node
        node.parent = left

        node.left = leftright
        leftright.parent = node

        node.update_height()
        left.update_height()

    def _rotate_rr(self, node: InternalAVL[T]) -> None:
        """
        Rotates the tree to the left (deepest on "right right" branch)

        Args:
            node (InternalAVL[T]): Node to rotate
        """
        assert isinstance(node.right, Internal)
        right = node.right
        rightleft = right.left

        if node is self._root:
            self._root = cast(I, right)
            right.parent = None
        else:
            right.parent = node.parent
            if node.parent.left is node:
                right.parent.left = right
            else:
                assert node.parent.right is node
                right.parent.right = right

        right.left = node
        node.parent = right

        node.right = rightleft
        rightleft.parent = node

        node.update_height()
        right.update_height()


class ScalarTreeAVL[T: SupportLT](TreeAVL[T, LeafAVL[T], InternalAVL[T]]):  
    """
    An AVL tree class for scalars

    This is an AVL tree class intended for simple scalars (e.g. int, float).
    In theory it could be used for more complex type, provided they have
    a less than comparison operator implemented. This class was only
    intended for testing purposes.
    """

    def insert(self, x: T) -> LeafAVL[T]:
        """
        Insert a scalar into the tree

        Args:
            x (T): Scalar to be inserted

        Return:
            LeafAVL[T]: The leaf node added to the tree
        """
        if self._root is None:
            self._root = LeafAVL(x)
            return self._root

        sibling = self._get_sibling(x)
        node = self._insert(x, sibling)
        #assert node.parent is not None
        self._update_internals(node)
        self._rebalance(node)
        return node

    def _get_sibling(self, x: T) -> LeafAVL[T]:
        """
        Get the sibling of the value to be inserted

        Args:
            x (T): Scalar to be inserted

        Return:
            LeafAVL[T]: Sibling node of the scalar to be inserted
        """
        sibling = self._root
        while isinstance(sibling, InternalAVL):
            if x < sibling.internal[0]:
                sibling = sibling.left
            else:
                sibling = sibling.right
        assert isinstance(sibling, LeafAVL)
        return sibling

    def _insert(self, x: T, sibling: LeafAVL[T]) -> LeafAVL[T]:
        """
        Insert a scalar next to its sibling

        Args:
            x (T): Scalar to be inserted
            sibling (LeafAVL[T]): Node where the new nodes will be created

        Return:
            LeafAVL[T]: The new leaf created with the input scalar value
        """
        node = LeafAVL(x)

        if x < sibling._value:
            internal = InternalAVL(x, sibling._value)
            internal.left = node
            internal.right = sibling
        else:
            internal = InternalAVL(sibling._value, x)
            internal.left = sibling
            internal.right = node

        node.parent = internal

        if sibling is self._root:
            internal.parent = None
            sibling.parent = internal
            self._root = internal
        else:
            internal.parent = sibling.parent
            sibling.parent = internal
            if internal.parent.left is sibling:
                internal.parent.left = internal
            else:
                assert internal.parent.right is sibling
                internal.parent.right = internal

        return node

    def _update_internals(self, node: LeafAVL[T]) -> None:
        """
        Updates successor and predecessor values of internals in response
        to the newly insert leaf

        Args:
            node (LeafAVL[T]): The newest leaf added; the place to start the updating
        """        
        if node is self._root or node.parent is self._root:
            return
        
        current = node.parent
        parent = current.parent
        if current.left is node:
            while True:
                if parent.right is current:
                    # successor = cast(LeafAVL[T], node.parent.right)
                    # assert parent.internal[1] == successor.value
                    parent.internal[1] = node.value
                    break
                if parent is self._root:
                    break
                current = parent
                parent = current.parent
        else:
            assert current.right is node
            while True:
                if parent.left is current:
                    # predecessor = cast(LeafAVL[T], current.left)
                    # assert parent.internal[0] == predecessor.value
                    parent.internal[0] = node.value
                    break
                if parent is self._root:
                    break
                current = parent
                parent = current.parent