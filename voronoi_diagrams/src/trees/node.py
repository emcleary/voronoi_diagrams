from __future__ import annotations

from abc import ABC, abstractmethod


class Node[L, I](ABC):
    """
    A node in a tree

    This is a base class for node objects used in trees.
    Leaf nodes contain data of type L, while Internal nodes
    contain data of type I. While the parent is initialized
    to None, it is assumed it will get set as needed before 
    calling the getter. Primarily asserting not None was 
    done simplify type hinting.

    Attributes:
        _parent (Internal or None): A parent node
    """
    def __init__(self):
        """
        Creates a node object
        """
        self._parent: Internal[L, I] | None = None

    @property
    def parent(self) -> Internal[L, I]:
        """
        Gets the parent node

        Getting the parent node assumes the parent is set,
        i.e. this node is not the root node of a tree.

        Return:
            Internal: A parent node
        """
        assert self._parent is not None
        return self._parent

    @parent.setter
    def parent(self, node: Internal[L, I] | None) -> None:
        """
        Sets the parent node

        Args:
            node (Internal or None): A parent node
        """
        self._parent = node

    @property
    @abstractmethod
    def height(self) -> int:
        ...

    @abstractmethod
    def update_height(self) -> None:
        ...

    @property
    @abstractmethod
    def imbalance(self) -> int:
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...


class Leaf[L, I](Node[L, I]):
    """
    A leaf in a tree

    This is a Leaf object containing data of type L.
    Any parent will be an Internal node containing data
    of type I.
    """
    def __init__(self, value: L):
        """
        Construct a Leaf object

        Args:
            value [L]: Data to store in the Leaf
        """
        super(Leaf, self).__init__()
        self._value = value

    @property
    def value(self) -> L:
        """
        Get the data stored in the Leaf

        Return:
            L: Data stored in the Leaf
        """
        return self._value

    @property
    def height(self):
        """
        Get the height of Leaf in the tree (always 0)
        """
        return 0
    
    def update_height(self):
        """
        Update the Leaf's height in the tree (never needed)
        """
        pass

    @property
    def imbalance(self) -> int:
        """
        Get the imbalance between children (always 0 in a Leaf)
        """
        return 0

class Internal[L, I](Node[L, I]):
    """
    An internal node in a tree

    Data stored in Internal nodes will be of type I, while any
    Leaf in the tree will be of type L. While constructed with
    no parent or children, it is assumed that at least the children
    will be set externally. Assertions they are set in getters was
    primarily done to simplify type hinting.

    Attributes:
        _internal: Data stored in the node
        _left [Leaf or Internal]: Left child
        _right [Leaf or Internal]: Right child
        _height [int]: Height of the node relative to its deepest child leaf
    """

    def __init__(self, internal: I):
        """
        Constructor for an Internal node

        Args:
            internal [I]: Data stored in the node
        """
        super(Internal, self).__init__()
        self._internal = internal
        self._left: Leaf[L, I] | Internal[L, I] | None = None
        self._right: Leaf[L, I] | Internal[L, I] | None = None
        self._height = 0

    # redundant with value property, but included anyway for a bit of clarity in the code
    @property
    def internal(self) -> I:
        """
        Get the internal data

        Return:
            I: Data stored in the node
        """
        return self._internal
    
    @property
    def value(self) -> I:
        """
        Get the internal data

        Return:
            I: Data stored in the node        
        """
        return self._internal

    @property
    def height(self) -> int:
        """
        Get the height of the node relative to its deepest leaf

        Return:
            int: Height of the node
        """
        assert self._height > 0
        return self._height

    def update_height(self) -> None:
        """
        Updates the height attribute with current children
        """
        assert self._left is not None
        assert self._right is not None
        self._height = 1 + max(self._left.height, self._right.height)

    @property
    def imbalance(self) -> int:
        """
        Get the imbalance between current children
        """
        return self.left.height - self.right.height

    @property
    def left(self) -> Leaf[L, I] | Internal[L, I]:
        """
        Get the left child
        
        Return:
            Leaf or Internal: The left child node
        """
        assert self._left is not None
        return self._left
    
    @left.setter
    def left(self, node: Leaf[L, I] | Internal[L, I]) -> None:
        """
        Set the left child

        Args:
            node [Leaf or Internal]: The left child
        """
        self._left = node

    @property
    def right(self) -> Leaf[L, I] | Internal[L, I]:
        """
        Get the right child
        
        Return:
            Leaf or Internal: The right child node
        """
        assert self._right is not None
        return self._right
    
    @right.setter
    def right(self, node: Leaf[L, I] | Internal[L, I]) -> None:
        """
        Set the right child

        Args:
            node [Leaf or Internal]: The right child
        """
        self._right = node

