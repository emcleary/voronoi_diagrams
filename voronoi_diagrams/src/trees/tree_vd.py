from __future__ import annotations

import numpy as np

from voronoi_diagrams.src.doubly_connected_edge_list import PointDCEL
from voronoi_diagrams.src.math_voronoi import get_parabola_intersection
from voronoi_diagrams.src.point import Point2D
from voronoi_diagrams.src.trees.tree_avl import TreeAVL, LeafAVL, InternalAVL


from typing import Tuple, List, cast, TYPE_CHECKING
if TYPE_CHECKING:
    from ..events import CircleEvent


class VoronoiEdge:
    """
    An object tracking the Voronoi edge

    Attributes:
        endpoints (List[PointDCEL]): Pair of endpoints of the Voronoi edge
    """
    def __init__(self):
        """
        Constructs Voronoi edge, defaulting endpoints to None
        """
        self.endpoints: List[PointDCEL | None] = [None, None]

    def is_closed(self) -> bool:
        """
        Check if both endpoints are set

        Return:
            bool: True if both endpoints are set, false otherwise
        """
        return self.endpoints[0] is not None \
            and self.endpoints[1] is not None

    def add_endpoint(self, point: PointDCEL) -> None:
        """
        Add an endpoint to the edge

        Args:
            point (PointDCEL): Endpoint to add
        """
        assert not self.is_closed()
        if self.endpoints[0] is None:
            self.endpoints[0] = point
        else:
            self.endpoints[1] = point

    def __str__(self) -> str:
        return '(' + ', '.join(map(str, self.endpoints)) + ')'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self)})'


class LeafVD(LeafAVL[Point2D]):
    """
    A leaf object for the Voronoi diagram tree

    A leaf object with typing specifically used for Voronoi diagram
    trees. Data inserted is of type Point2D. The tree class is derived
    from the AVL tree, using the same Leaf and Internal types. 

    Attributes:
        _circle (Circle Event or None): A circle event belonging to this node, if any
    """

    def __init__(self, point: Point2D):
        """
        Construct a LeafVD object

        Args:
            point (P): A 2-dimensional point for the node data
        """
        super(LeafVD, self).__init__(point)
        self._circle: CircleEvent | None = None

    def deactivate_circle(self) -> None:
        """
        Deactivates the circle event, if present
        """
        if self._circle:
            self._circle.deactivate()
            self._circle = None

    @property
    def circle(self) -> CircleEvent | None:
        """
        Getter for the circle event 

        Return:
            CircleEvent or None: The circle event, if present, otherwise None
        """
        return self._circle


class InternalVD(InternalAVL[Point2D]):
    """
    An internal object used for Voronoi diagram trees

    An internal object with typing specifically used for Voronoi diagram
    trees. Data inserted is of type Point2D. The tree class is derived
    from the AVL tree, using the same Leaf and Internal types.

    Attributes:
        edge (VoronoiEdge): A pair of edge endpoints built by this node
    """

    def __init__(self, p0: Point2D, p1: Point2D, endpoints: VoronoiEdge):
        """
        Construct an InternalVD object

        Args:
            p0 (Point2D): The focus of a parabola used as an internal value
            p1 (Point2D): The focus of a parabola used as an internal value
            endpoints (VoronoiEdge): The edge object maintained by this node
        """
        super(InternalVD, self).__init__(p0, p1)
        self.edge = endpoints

    def calculate_breakpoint(self, directrix: float) -> Tuple[float, float]:
        """
        Calcluates a breakpoint at a sweepline value

        Args:
            directrix (float): The sweepline value, seen as a parabola parameter

        Return:
            Tuple[float, float]: x, y coordinates of the breakpoint
        """
        pa, pb = self.internal
        return get_parabola_intersection(pa, pb, directrix)

    def __str__(self) -> str:
        src, dest = self.internal
        return str(f'{src} <--> {dest}')



class TreeVD(TreeAVL[Point2D, LeafVD, InternalVD]):
    """
    A tree for the Voronoi diagram balanced with AVL methods

    This class is used for Fortune's algorithm for building 
    Voronoi diagrams. Each leaf is a 2 dimension points representing
    an inserted site. Each internal node contains a pair of sites that serve
    as foci for parabolas. The y-coordinate of each newly inserted
    point serves as a sweepline value. As the algorithm progresses, the
    sweepline moves upward. When a leaf gets deleted, so does a pair of
    internal nodes. Their edges will have an endpoint set (externally).

    Attributes:
        _colinear_points (bool): Tracks if all inserted points are colinear
        _colinear_nodes (List[InternalVD]): Tracks internals nodes generated from
            initial colinear points that are omitted from the tree itself and must
            be postprocessed separately.
    """
    def __init__(self):
        """
        Constructs a TreeVD object
        """
        super(TreeVD, self).__init__()
        self._colinear_points = True
        self._colinear_nodes: List[InternalVD] = []

    def insert(self, x: Point2D) -> LeafVD:
        """
        Insert a 2-dimensional point

        Args:
            x (Point2D): A site to be inserted

        Return:
            LeafVD: A leaf node created for the site
        """
        if self._root is None:
            self._root = LeafVD(x)
            return self._root

        sibling = self._get_sibling(x)
        node = self._insert(x, sibling)
        self._rebalance(node)
        return node

    def _get_sibling(self, point: Point2D) -> LeafVD:
        """
        Get the sibling of the value to be inserted

        Args:
            point (Point2D): The site to be inserted

        Return:
            LeafVD: The sibling node where the new site should be inserted
        """
        assert self._root is not None
        if self._colinear_points:
            node = self._root
            while True:
                if isinstance(node, LeafVD):
                    if np.isclose(point.y, node._value.y) and np.greater(point.x, node._value.x):
                        return node
                elif isinstance(node, InternalVD):
                    if np.isclose(point.y, node._internal[1].y) and np.greater(point.x, node._internal[1].x):
                        node = node.right
                        continue
                self._colinear_points = False
                break

        node = self._root
        while isinstance(node, InternalVD):
            x, _ = node.calculate_breakpoint(point.y)
            if np.isclose(point.x, x):
                # either left or right should be fine here
                node = node.left
            elif point.x < x:
                node = node.left
            else:
                node = node.right

        assert isinstance(node, LeafVD)
        return node


    def _insert(self, point: Point2D, sibling: LeafVD) -> LeafVD:
        """
        Insert a site next to its sibling

        Args:
            point (Point2D): Site to be inserted
            sibling (LeafVD): Node where the new nodes will be created

        Return:
            LeafVD: The new leaf created with the input site
        """
        if self._colinear_points and sibling.value.y == point.y:
            pi = point
            pj = sibling.value
            assert pj.x < pi.x
            assert not np.isclose(pj.x, pi.x)
            voronoi_edge = VoronoiEdge()
            internal = InternalVD(pj, pi, voronoi_edge)
            internal_twin = InternalVD(pi, pj, voronoi_edge)
            self._colinear_nodes.append(internal_twin)

            if sibling is self._root:
                self._root = internal
            else:
                internal.parent = sibling.parent
                if internal.parent.left is sibling:
                    internal.parent.left = internal
                else:
                    assert internal.parent.right is sibling
                    internal.parent.right = internal
            
            internal.left = LeafVD(pj)
            internal.left.parent = internal
            internal.right = LeafVD(pi)
            internal.right.parent = internal
            return internal.right

        pi = point
        pj = sibling._value
        voronoi_edge = VoronoiEdge()
        internal_left = InternalVD(pj, pi, voronoi_edge)
        internal_right = InternalVD(pi, pj, voronoi_edge)

        if sibling is self._root:
            self._root = internal_right
        else:
            internal_right.parent = sibling.parent
            if internal_right.parent.left is sibling:
                internal_right.parent.left = internal_right
            else:
                assert internal_right.parent.right is sibling
                internal_right.parent.right = internal_right

        # an existing circle is a fake event
        # it no longer exists because a new point begin added would be inside this circle
        sibling.deactivate_circle()
        del sibling

        node_right = LeafVD(pj)
        node_center = LeafVD(pi)
        node_left = LeafVD(pj)

        internal_right.left = internal_left
        internal_right.left.parent = internal_right

        internal_right.right = node_right
        internal_right.right.parent = internal_right

        internal_left.right = node_center
        internal_left.right.parent = internal_left

        internal_left.left = node_left
        internal_left.left.parent = internal_left

        return node_center

    def delete_node(self, node: LeafVD) -> Tuple[InternalVD, InternalVD, InternalVD]:
        """
        Deletes a leaf node and corresponding internal nodes from the tree

        Args:
            node (LeafVD): Leaf node to be deleted

        Return:
            Tuple[InternalVD, InternalVD, InternalVD]: The new internal node created
                along with the left and right internal nodes deleted
        """
        assert node is not self._root
        # in this application, no node will be deleted until there are
        # at least 3 points inserted; at this point the root should be
        # an internal node

        if node.parent.left is node:
            replacement = self.get_successor(node)
            assert replacement is not None
            internal_right = node.parent
            right = node.parent.right
            assert right.parent is internal_right

            ### IMPOSSIBLE for internal_right to be the root
            # so its parent will exist
            assert internal_right is not self._root

            ### REMOVE INTERNAL_RIGHT
            right.parent = internal_right.parent
            if right.parent.right is internal_right:
                right.parent.right = right
            else:
                assert right.parent.left is internal_right
                right.parent.left = right

            ### FIND INTERNAL_LEFT
            current = right
            while current.parent.left is current:
                current = current.parent
            assert current is not self._root
            assert current.parent._internal[1] is node._value
            internal_left = current.parent

            ### CREATE NEW INTERNAL NODE
            voronoi_edge = VoronoiEdge()
            internal_new = InternalVD(internal_left._internal[0], replacement._value, voronoi_edge)

            ### REPLACE INTERNAL_LEFT WITH NEW NODE
            internal_new.left = internal_left.left
            internal_new.left.parent = internal_new
            internal_new.right = internal_left.right
            internal_new.right.parent = internal_new
            if internal_left is self._root:
                self._root = internal_new
            else:
                internal_new.parent = internal_left.parent
                if internal_new.parent.left is internal_left:
                    internal_new.parent.left = internal_new
                else:
                    assert internal_new.parent.right is internal_left
                    internal_new.parent.right = internal_new

        else:
            replacement = self.get_predecessor(node)
            assert replacement is not None
            internal_left = node.parent
            left = node.parent.left
            assert left.parent is internal_left

            ### IMPOSSIBLE for internal_left to be the root
            # so its parent will exist
            assert internal_left is not self._root

            ### REMOVE INTERNAL_RIGHT
            parent = internal_left.parent
            if parent.right is internal_left:
                parent.right = left
            else:
                assert parent.left is internal_left
                parent.left = left
            left.parent = parent

            ### FIND INTERNAL_RIGHT
            current = left
            while current.parent.right is current:
                current = current.parent
            assert current is not self._root
            assert current.parent._internal[0] is node._value
            internal_right = current.parent

            ### CREATE NEW INTERNAL NODE
            voronoi_edge = VoronoiEdge()
            internal_new = InternalVD(replacement._value, internal_right._internal[1], voronoi_edge)

            ### REPLACE INTERNAL_RIGHT WITH NEW NODE
            internal_new.left = internal_right.left
            internal_new.left.parent = internal_new
            internal_new.right = internal_right.right
            internal_new.right.parent = internal_new
            if internal_right is self._root:
                self._root = internal_new
            else:
                internal_new.parent = internal_right.parent
                if internal_new.parent.left is internal_right:
                    internal_new.parent.left = internal_new
                else:
                    assert internal_new.parent.right is internal_right
                    internal_new.parent.right = internal_new

        # for this application, it should be impossible to delete
        # either the furthest left or furthest right leaves
        assert internal_left and internal_right

        self._rebalance(replacement)

        return internal_new, \
            cast(InternalVD, internal_left), \
            cast(InternalVD, internal_right)
