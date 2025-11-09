from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from voronoi_diagrams.src.edge import Edge
from voronoi_diagrams.src.point import Point2D
from voronoi_diagrams.src.trees.tree_bvh import BalancedTreeBVH, TreeBVH

from typing import Self, List


class PointDCEL(Point2D):
    """
    A 2-dimensional point for the doubly-connected edge list

    Attributes:
        edge (EdgeDCEL): An edge containing this point as its source
    """

    def __init__(self, x: float, y: float):
        """
        Creates a 2-dimensional point for the doubly-connected edge list

        Args:
            x (float): x-coordinate
            y (float): y-coordinate
        """
        super(PointDCEL, self).__init__(x, y)
        self.edge: EdgeDCEL | None = None


class EdgeDCEL(Edge[PointDCEL]):
    """
    An edge joining two 2-dimensional points in the doubly-connected edge list

    Attributes:
        _twin (EdgeDCEL): This object's flipped edge
        _next (EdgeDCEL): The next edge counterclockwise around the cell
        _prev (EdgeDCEL): The next edge clockwise around the cell
    """

    def __init__(self, source: PointDCEL, destination: PointDCEL):
        """
        Creates an edge for the doubly-connected edge list

        Args:
            source (PointDCEL): The source point
            destination (PointDCEL): The destination point
        """
        super(EdgeDCEL, self).__init__(source, destination)
        self._twin: Self | None = None
        self._next: Self | None = None
        self._prev: Self | None = None

    @property
    def twin(self) -> Self:
        """
        Get this edge's twin

        The twin is assumed to have been set right after construction,
        hence the assertion.

        Return:
            EdgeDCEL: This edge's twin
        """
        assert self._twin is not None
        return self._twin

    @twin.setter
    def twin(self, edge: Self) -> None:
        """
        Set this edge's twin

        An edge and its twin must use the same objects for their
        source and destinations, hence the assertion checks.

        Args:
            edge (EdgeDCEL): This edge's twin
        """
        assert self.src is edge.dest
        assert self.dest is edge.src
        self._twin = edge

    @property
    def next(self) -> Self:
        """
        Get this edge's next edge

        Return:
            EdgeDCEL: This edge's next edge
        """
        assert self._next is not None
        return self._next

    @next.setter
    def next(self, edge: Self) -> None:
        """
        Set this edge's next edge

        The destination of this edge must be the source of
        the next edge, hence the assertion check.

        Args:
            edge (EdgeDCEL): This edge's next edge
        """
        assert self.dest is edge.src
        self._next = edge

    @property
    def prev(self) -> Self:
        """
        Get this edge's prev edge

        Return:
            EdgeDCEL: This edge's prev edge
        """
        assert self._prev is not None
        return self._prev

    @prev.setter
    def prev(self, edge: Self) -> None:
        """
        Set this edge's prev edge

        The source of this edge must be the destination of
        the previous edge, hence the assertion check.

        Args:
            edge (EdgeDCEL): This edge's prev edge
        """
        assert self.src is edge.dest
        self._prev = edge

    def rotate(self) -> Self:
        """
        Get the next edge counterclockwise around this edge's source point

        Return:
            EdgeDCEL: The next rotated edge
        """
        return self.prev.twin


class DoublyConnectedEdgeList:
    """
    Stores vertices and edges as a doubly-connected edge list

    Attributes:
        _vertex_tree (TreeBVH[PointDCEL] or BalancedTreeBVH[PointDCEL]): A tree for managing vertices
        _vertices (List[PointDCEL]): A list of vertices
        _edges (List[EdgeDCEL]): A list of (half-)edges
        _shortest_edge_length (float): Shortest length of edges
        _longest_edge_length (float): Longest length of edges
    """

    def __init__(self, balance_vertex_tree: bool = False):
        """
        Initializes the doubly-connected edge list

        Args:
            balanced_vertex_tree (bool): Use a balanced BVH tree for vertices if true, otherwise use an unbalanced tree
        """
        self._vertex_tree = BalancedTreeBVH[PointDCEL]() if balance_vertex_tree else TreeBVH[PointDCEL]()
        self._vertices: List[PointDCEL] = []
        self._edges: List[EdgeDCEL] = []
        self._shortest_edge_length = np.inf
        self._longest_edge_length = 0.0

    @property
    def shortest_edge_length(self) -> float:
        """
        Get the shortest edge length

        Return:
            float: Shortest edge length
        """
        return self._shortest_edge_length

    @property
    def longest_edge_length(self) -> float:
        """
        Get the longest edge length

        Return:
            float: Longest edge length
        """
        return self._longest_edge_length

    def get_closest_vertex(self, point: Point2D, radius: float = 1e-8) -> PointDCEL | None:
        """
        Get the vertex at the given point, if any

        This method searches the vertex tree to see if it contains the point
        or another point to within some error. It returns the point if found,
        otherwise None.

        Args:
            point (Point2D): Point to search for in the tree
            radius (float, optional): A tolerance for searching for the point, default 1e-8

        Return:
            PointDCEL or None: Return a point if any are found, otherwise None
        """
        node = self._vertex_tree.query(point, radius)
        if node is not None:
            return node.value
        return None

    def add_vertex(self, point: Point2D, radius=1e-8) -> PointDCEL:
        """
        Add a point to the vertex tree

        This adds a point to the vertex tree. An assertion checks to
        ensure neither the point nor any nearby point already exists
        in the tree.

        Args:
            point (Point2D): Point to copy into the vertex tree

        Return:
            PointDCEL: Point added to the tree
        """
        assert self.get_closest_vertex(point, radius) == None
        vertex = PointDCEL(*point._x)
        self._vertex_tree.insert(vertex)
        self._vertices.append(vertex)
        return vertex

    def create_edge(self, src: PointDCEL, dest: PointDCEL) -> EdgeDCEL | None:
        """
        Creates an edge from 2 vertices

        This creates and edge joining 2 vertices. The source point is expected to
        be in the vertex tree; the destination point may or may not be.
        Additionally, both points must be unique, otherwise no edge will be added.

        The "_next" attributes of edges serves a temporary linked list
        for all edges sharing a source vertex. See the "postprocess" method
        for how the "_next" and "_prev" attributes are finalized.

        Args:
            src (PointDCEL): A point for the edge's source
            dest (PointDCEL): A point for the edge's destination

        Return:
            EdgeDCEL: An edge joining the source to the destination
        """

        # check that the source point is in the tree
        # destination point not required to be in the tree as it can
        # be the "infinity" point of a Voronoi diagram; not adding these
        # destination points to the tree can speed up postprocessing
        assert self._vertex_tree.contains(src)
        # assert self._vertex_tree.contains(dest)

        # don't add an edge if they are the same point
        # e.g. can happen in VD algorithm if >3 points are
        # on a circle
        if src is dest:
            return None

        # double check to ensure the points are distinct
        assert src != dest

        e01 = EdgeDCEL(src, dest)
        e10 = EdgeDCEL(dest, src)

        e01.twin = e10
        if src.edge is not None:
            e01._next = src.edge
        src.edge = e01

        e10.twin = e01
        if dest.edge is not None:
            e10._next = dest.edge
        dest.edge = e10

        self._edges.append(e01)
        self._edges.append(e10)

        self._shortest_edge_length = min(self._shortest_edge_length, src.distance(dest))
        self._longest_edge_length = max(self._longest_edge_length, src.distance(dest))

        return e01

    def postprocess(self) -> None:
        """
        Postprocesses the doubly-connected edge list

        This method postprocesses the edge list to set all "_next"
        and "_prev" edge attributes correctly. Before postprocessing,
        "_prev" attributes are not set, and the "_next" attributes are
        a linked list of all edges sharing the source vertex. Worst case
        complexities are O(n log n) for time and O(n) for memory, typically
        seen when a vertex has a high degree (e.g. VD with all sites are on
        a circle). Typically this will have linear complexity in time and
        constant in space.
        """

        for vertex in self._vertices:
            edge = vertex.edge
            assert edge is not None

            # list edges from existing linked list ("_next")
            edges = []
            while edge._next is not None:
                edges.append(edge)
                edge = edge.next
            edges.append(edge)

            # sort edges (counterclockwise about source)
            center = vertex
            def calc_theta(edge: EdgeDCEL) -> float:
                radius = center.distance(edge.dest)
                dx = edge.dest.x - center.x
                dy = edge.dest.y - center.y
                cosine = dx / radius
                if dy > 0 or  np.isclose(dy, 0):
                    return np.arccos(cosine)
                return 2*np.pi - np.arccos(cosine)

            edges.sort(key=calc_theta)

            # keep the unsorted linked list in tact for now ("_next");
            # use sorted list to update "_prev" only
            for edge0, edge1 in zip(edges, edges[1:]):
                edge0.prev = edge1.twin
            edges[-1].prev = edges[0].twin

        # now finalize "_next" using "_prev"
        for edge in self._edges:
            if edge._prev is None:
                # This edge's src is an "infinity" point in the VD
                # Ensure this edge's dest joined to another edge
                assert edge.twin._prev is not None
                edge.prev = edge.twin
            edge.prev.next = edge

    def plot(self, show_vertices=False, show=True, xlim=None, ylim=None) -> Figure:
        """
        Plot edges from the edge list

        Args:
            show_vertices (bool, optional): Include vertex points, default false
            show (bool, optional): Show the plot, default true
            xlim (Tuple[float, float], optional): Set limits of the x-axis, default None
            ylim (Tuple[float, float], optional): Set limits of the y-axis, default None
        """
        fig = plt.figure()
        ax = fig.add_subplot()

        for edge in self._edges:
            ax.plot([edge.src.x, edge.dest.x], [edge.src.y, edge.dest.y], 'k')

        if show_vertices:
            for vertex in self._vertices:
                ax.plot(vertex.x, vertex.y, 'r', marker='o', ms=8)

        if xlim:
            ax.set_xlim(*xlim)

        if ylim:
            ax.set_ylim(*ylim)

        if show:
            fig.show()

        return fig
