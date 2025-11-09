from __future__ import annotations

import heapq
import numpy as np

from voronoi_diagrams.src.point import Point
from voronoi_diagrams.src.trees.aabb import AABB
from voronoi_diagrams.src.trees.node import Leaf, Internal
from voronoi_diagrams.src.trees.tree import Tree

from typing import List, Tuple, Protocol, cast


class NodeBVH(Protocol):
    """
    A protocol to help for arguments that can be either
    LeafBVH or InternalBVH
    """
    @property
    def box(self) -> AABB:
        ...

class InternalBVH[P: Point](Internal[P, AABB]):
    """
    An internal object for the BVH tree

    An internal object for typing specifically with the BVH tree.
    Type P can be a Point or any of its derived classes. This
    also serves as the data in the LeafBVH object. Internal nodes
    store an AABB object for data.

    Attributes:
        _count (int): Number of nodes in the subtree
    """

    def __init__(self, ndim: int):
        """
        Constructs an InternalBVH object

        This constructs an InternalBVH object. Initially the object's
        child nodes are set to None. They must be set manually,
        after which the data (AABB) can be properly set.

        Args:
            ndim (int): Number of dimensions of the bounding box
        """
        super(InternalBVH, self).__init__(AABB(ndim=ndim))
        self._count = 0

    @property
    def box(self) -> AABB:
        """
        Getter for the bounding box (_internal)

        Return:
            AABB: The box containing the leaf data point only
        """
        return self._internal

    def set_box(self) -> None:
        """
        Set the bounding box using child nodes
        """
        assert isinstance(self.left, LeafBVH | InternalBVH)
        assert isinstance(self.right, LeafBVH | InternalBVH)
        self.box._set(self.left.box._pmin, self.left.box._pmax,
                       self.right.box._pmin, self.right.box._pmax)
    
    @property
    def count(self) -> int:
        """
        Getter for the count attribute

        Return:
            int: Number of nodes in the subtree
        """
        return self._count
    
    def update_count(self) -> None:
        """
        Updates the count attribute using child nodes
        """
        assert isinstance(self.left, LeafBVH | InternalBVH)
        assert isinstance(self.right, LeafBVH | InternalBVH)
        self._count = 1 + self.left.count + self.right.count

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.box._pmin}, {self.box._pmax}))" \
            + f"\nc={self._count} h={self._height} id={id(self)}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.box._pmin}, {self.box._pmax}))"

    def __lt__(self, other: LeafBVH[P] | InternalBVH[P]):
        if isinstance(other, LeafBVH):
            return self.box._pmin < other.value
        assert isinstance(other, InternalBVH)
        return self.box < other.box

class LeafBVH[P: Point](Leaf[P, AABB]):
    """
    A leaf object for the BVH tree

    A leaf object for typing specifically with the BVH tree.
    Type P can be a Point or any of its derived classes. This
    also serves as the data in the LeafBVH object. Internal nodes
    store an AABB object for data.

    Attributes:
        _box (AABB): A box containing the data point
        _count (int): Number of nodes in the subtree (always 1 for a leaf)
    """

    def __init__(self, point: P):
        """
        Construct a LeafBVH object

        Args:
            point (P): A point for the node data
        """
        super(LeafBVH, self).__init__(point)
        self._box = AABB(point)
        self._count = 1

    @property
    def box(self) -> AABB:
        """
        Getter for the box attribute

        Return:
            AABB: The box containing the leaf data point only
        """
        return self._box

    @property
    def count(self) -> int:
        """
        Getter for the count attribute

        Return:
            int: Number of nodes in the subtree
        """
        return self._count

    def update_count(self) -> None:
        """
        Updates the count attribute (does nothing for the LeafBVH)
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._value})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._value})"

    def __lt__(self, other: LeafBVH[P] | InternalBVH[P]):
        if isinstance(other, InternalBVH):
            return self.value < other.box._pmin
        assert isinstance(other, LeafBVH)
        return self.value < other.value



class TreeBVH[P: Point](Tree[P, LeafBVH[P], InternalBVH[P]]):
    """
    A Boundary Volume Hierarchy (BVH) tree

    This tree is a Boundary Volume Hierarchy (BVH) tree, implemented
    as a online algorithm for the purpose of tracking vertex points
    and looking them up efficiently in the doubly-connected edge tree
    (expect average O(log n) time complexity vs linear complexity for just
    a list). As a result, it can prevent degenerate point from being added.

    References:

    Omohundro, Stephen M. Five Balltree Construction Algorithms. (1989)
    https://www.academia.edu/download/80186366/tr-89-063.pdf
    
    Erin Catto. Dynamic Boundary Volume Hierarchies. (2019)
    https://box2d.org/files/ErinCatto_DynamicBVH_GDC2019.pdf

    """

    def get_bounds(self) -> Tuple[P, P]:
        """
        Getter for the bounds of the tree's points

        Return:
            Tuple[P, P]: Minimum and maximum bound points
        """
        assert self._root is not None
        return self._root.box._pmin, self._root.box._pmax
    
    def insert(self, x: P) -> LeafBVH[P]:
        """
        Insert a point into the tree

        Args:
            x (P): Point to insert

        Return:
            LeafBVH: New node inserted
        """
        if self._root is None:
            self._root = LeafBVH(x)
            return self._root

        node = self._insert(x)
        self._update_internals(node)
        return node

    def _insert(self, point: P) -> LeafBVH:
        """
        Insert a point into the tree

        Args:
            point (P): Point to insert

        Return:
            LeafBVH: New node inserted
        """
        assert self._root is not None

        def get_surface_area(node: NodeBVH) -> Tuple[float, float]:
            sa = node.box.surface_area
            point_sa = node.box.proposed_surface_area(point)
            return sa, point_sa

        queue: List[Tuple[float, LeafBVH[P] | InternalBVH[P]]] = [(0, self._root)]
        best_cost = np.inf
        best_node = None

        while queue:
            inherited_cost, node = heapq.heappop(queue)
            sa, point_sa = get_surface_area(node)
            delta_sa = point_sa - sa
            node_cost = sa + inherited_cost
            ### TODO: test if "=" or "<=" is better for performance
            ### presumably "=" is better; first node seen with specific cost
            ### will be used; first node is more likely to be shallower than later nodes;
            ### DEFINITELY not always true
            if node_cost <= best_cost: # equal to give priority to nodes DEEPER in the tree
                best_cost = node_cost
                best_node = node
            # OMITTING cost of point here (assumed 0); would be positive if a shape or if adding padding
            low_cost = inherited_cost + delta_sa
            if low_cost < best_cost:
                if isinstance(node, InternalBVH):
                    assert isinstance(node.left, LeafBVH | InternalBVH)
                    assert isinstance(node.right, LeafBVH | InternalBVH)
                    heapq.heappush(queue, (low_cost, node.left))
                    heapq.heappush(queue, (low_cost, node.right))

        assert best_node is not None
        sibling = best_node

        newnode = LeafBVH(point)
        internal = InternalBVH(point.dimension)
        if sibling is self._root:
            self._root = internal
        else:
            internal.parent = sibling.parent
            if internal.parent.left is sibling:
                internal.parent.left = internal
            else:
                assert internal.parent.right is sibling
                internal.parent.right = internal

        internal.left = sibling
        internal.right = newnode
        internal.left.parent = internal
        internal.right.parent = internal
        internal.set_box()
        internal.update_count()
        internal.update_height()

        return newnode

    def _update_internals(self, leaf: LeafBVH[P]) -> None:
        """
        Update internal nodes in response to a new leaf

        Args:
            leaf (LeafBVH[P]): Newly inserted leaf where the update starts        
        """
        assert isinstance(self._root, InternalBVH)
        node = cast(InternalBVH, leaf.parent)
        box = node.box
        while node is not self._root:
            node = cast(InternalBVH, node.parent)
            node.box.union(box)
            node.update_count()
            node.update_height()
            box = node.box

    def query(self, point: Point, radius: float = 0) -> LeafBVH[P] | None:
        """
        Find a node containing or a near a given point

        Args:
            point (Point): The test point
            radius (float): Allowed distance from test point

        Return:
            LeafBVH[P] or None: Leaf near test point, if it exists        
        """
        if self._root is None:
            return None

        if radius == 0:
            box = AABB(point)
        else:
            pmin = Point(*(point._x - radius))
            pmax = Point(*(point._x + radius))
            box = AABB(pmin, pmax)

        stack: List[Leaf[P, AABB] | Internal[P, AABB]] = [self._root]
        while stack:
            node = stack.pop()
            if isinstance(node, LeafBVH):
                if np.less_equal(point.distance(node.value), radius):
                    return node
            else:
                assert isinstance(node, InternalBVH)
                if box.intersect(node.value):
                    stack.append(node.left)
                    stack.append(node.right)

        return None

    def contains(self, point: Point) -> bool:
        """
        Check if the tree contains this point object

        Args:
            point (Point): The test object

        Return:
            bool: True if the test object is in the tree, false otherwise
        """
        if self._root is None:
            return False

        stack: List[Leaf[P, AABB] | Internal[P, AABB]] = [self._root]
        while stack:
            node = stack.pop()
            if isinstance(node, LeafBVH):
                if node.value is point:
                    return True
            else:
                assert isinstance(node, InternalBVH)
                if node.box.contains(point):
                    stack.append(node.left)
                    stack.append(node.right)

        return False


class BalancedTreeBVH[P: Point](TreeBVH[P]):

    def insert(self, x: P) -> LeafBVH:
        """
        Insert a point with rebalancing

        Args:
            x (P): A point to be inserted

        Return:
            LeafBVH: New leaf containing the point
        """
        node = super(BalancedTreeBVH, self).insert(x)
        self._rebalance(node)
        return node

    def _rebalance(self, leaf: LeafBVH[P]) -> None:
        """
        Rebalances the tree

        Rebalances the tree by trying up to 4 possible swaps, and
        picking the tree with smallest cost.

        Args:
            leaf (LeafBVH[P]): New leaf where the rebalancing should start
        """
        if leaf is self._root:
            return
        
        def update(node: InternalBVH) -> None:
            node.update_count()
            node.update_height()

        def calculate_cost(node: InternalBVH) -> float:
            # scaling with counts make very little difference
            ratio = node.left.count / node.count
            cost = ratio * node.left.box.surface_area
            assert isinstance(node.right, LeafBVH | InternalBVH)
            ratio = node.right.count / node.count
            cost += ratio * node.right.box.surface_area
            # imbalance make a huge difference
            dc = max(1, abs(node.imbalance))
            assert isinstance(node.left, LeafBVH | InternalBVH)
            return cost * dc

        node = cast(InternalBVH, leaf.parent)        
        while True:

            dc = node.imbalance
            if abs(dc) >= 2:
                cost = np.inf
            else:
                cost = calculate_cost(node)

            left = node.left
            right = node.right

            cost_ll_r = np.inf
            cost_lr_r = np.inf
            cost_rr_l = np.inf
            cost_rl_l = np.inf

            if isinstance(left, InternalBVH):
                left_left = left.left
                left_right = left.right

                # swap right and left_left
                node.right = left_left
                left_left.parent = node
                left.left = right
                right.parent = left
                left.set_box()
                update(left)
                cost_ll_r = calculate_cost(node)
                # now swap back to default
                node.right = right
                right.parent = node
                left.left = left_left
                left_left.parent = left

                # swap right and left_right
                node.right = left_right
                left_right.parent = node
                left.right = right
                right.parent = left
                left.set_box()
                update(left)
                cost_lr_r = calculate_cost(node)
                node.right = right
                right.parent = node
                left.right = left_right
                left_right.parent = left

                left.set_box()
                update(left)

            if isinstance(right, InternalBVH):
                right_left = right.left
                right_right = right.right

                # swap left and right_left
                node.left = right_left
                right_left.parent = node
                right.left = left
                left.parent = right
                right.set_box()
                update(right)
                cost_rl_l = calculate_cost(node)
                # now swap back to default
                node.left = left
                left.parent = node
                right.left = right_left
                right_left.parent = right

                # swap left and right_right
                node.left = right_right
                right_right.parent = node
                right.right = left
                left.parent = right
                right.set_box()
                update(right)
                cost_rr_l = calculate_cost(node)
                # now swap back to default
                node.left = left
                left.parent = node
                right.right = right_right
                right_right.parent = right

                right.set_box()
                update(right)

            min_cost = min(cost, cost_ll_r, cost_lr_r, cost_rl_l, cost_rr_l)
            if min_cost == cost:
                pass
            elif min_cost == cost_ll_r:
                assert isinstance(node.left, InternalBVH)
                node.left.left, node.right = node.right, node.left.left
                node.left.left.parent = node.left
                node.right.parent = node
                node.left.set_box()
                update(node.left)
            elif min_cost == cost_lr_r:
                assert isinstance(node.left, InternalBVH)
                node.left.right, node.right = node.right, node.left.right
                node.left.right.parent = node.left
                node.right.parent = node
                node.left.set_box()
                update(node.left)
            elif min_cost == cost_rl_l:
                assert isinstance(node.right, InternalBVH)
                node.right.left, node.left = node.left, node.right.left
                node.right.left.parent = node.right
                node.left.parent = node
                node.right.set_box()
                update(node.right)
            elif min_cost == cost_rr_l:
                assert isinstance(node.right, InternalBVH)
                node.right.right, node.left = node.left, node.right.right
                node.right.right.parent = node.right
                node.left.parent = node
                node.right.set_box()
                update(node.right)

            node.update_height()
            if node is self._root:
                break
            node = cast(InternalBVH, node.parent)
