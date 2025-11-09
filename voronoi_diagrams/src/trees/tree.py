from abc import ABC, abstractmethod
import graphviz

from voronoi_diagrams.src.trees.node import Node, Leaf, Internal

from typing import Generator, List, cast


class Tree[T, L: Leaf, I: Internal](ABC):
    """
    A base class for Trees

    This is a base class for trees contain common methods used throughout,
    aside from distinct methods like insertion. L represents the type of the
    leaf object, while I represents the type of the Internal object.
    T represents the type of objects to be inserted.

    Attributes:
        _root (L, I, or None): The root node of a tree
        _plot_counter (int): Number of times the tree has been plotted (used for filenames)
    """
    
    _plot_counter = 1

    def __init__(self):
        """
        Creates a tree object, defaulting root to None
        """
        self._root: L | I | None = None

    @abstractmethod
    def insert(self, x: T) -> L:
        ...

    def get_successor(self, node: L | I) -> L | None:
        """
        Get the successor of a node

        Args:
            node (L or I): The input node

        Return:
            L or None: The successor of the argument node if it exists, None otherwise
        """
        if isinstance(node, Leaf):
            if node is self._root:
                return None

            current: Node = node
            while current.parent is not self._root and current.parent.right is current:
                current = current.parent

            if current.parent is self._root and current.parent.right is current:
                return None
            assert current.parent.left is current
            current = current.parent.right
        else:
            current = node.right

        while not isinstance(current, Leaf):
            current = current.left

        return cast(L, current)

    def get_predecessor(self, node: L | I) -> L | None:
        """
        Get the predecessor of a node

        Args:
            node (L or I): The input node

        Return:
            L or None: The predecessor of the argument node if it exists, None otherwise
        """
        if isinstance(node, Leaf):
            if node is self._root:
                return None

            current: Node = node
            while current.parent is not self._root and current.parent.left is current:
                current = current.parent

            if current.parent is self._root and current.parent.left is current:
                return None

            assert current.parent.right is current
            current = current.parent.left
        else:
            current = node.left

        while not isinstance(current, Leaf):
            current = current.right

        return cast(L, current)

    def get_leaves(self) -> Generator[L]:
        """
        Get the leaves of a tree

        Return:
            Generator[L]: A generate for leaves of the tree
        """
        if self._root is None:
            return
        
        stack: List[Node] = [self._root]
        while stack:
            node = stack.pop()
            if isinstance(node, Internal):
                stack.append(node.left)
                stack.append(node.right)
            else:
                assert isinstance(node, Leaf)
                yield cast(L, node)

    def get_internals(self) -> Generator[I]:
        """
        Get the internal nodes of a tree

        Return:
            Generator[I]: A generate for internal nodes of the tree
        """
        if self._root is None:
            return
        
        stack: List[Node] = [self._root]
        while stack:
            node = stack.pop()
            if isinstance(node, Internal):
                yield cast(I, node)
                stack.append(node.left)
                stack.append(node.right)

    def plot(self, filename: str | None = None,
             view: bool = False, image_format: str = 'png') -> None:
        """
        Plot the tree as a graph

        Args:
            filename (str, optional): Filename used to save the image
            view (bool, optional): Display the graph interactively if true
            image_format (str, optional): Image format to be used, png by default
        """

        if self._root is None:
            return

        graph = graphviz.Digraph('G')

        def to_name(node: Node) -> str:
            return str(id(node))

        def add_node(node: Node) -> str:
            name = to_name(node)
            label = str(node)
            graph.node(name, label)
            return name

        assert self._root is not None
        add_node(self._root)
        for node in self.get_internals():
            np = to_name(node)
            nl = add_node(node.left)
            nr = add_node(node.right)
            graph.edge(np, nl, arrowhead='none')
            graph.edge(np, nr, arrowhead='none')
        
        graph.format = image_format

        if view is True:
            graph.view()
            return

        if filename is None:
            filename = f'plot_{self.__class__._plot_counter}'
            self.__class__._plot_counter += 1

        graph.render(filename, view=False, cleanup=True)
        print('Plot written to', filename)
