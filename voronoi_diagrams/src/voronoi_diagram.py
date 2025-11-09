import heapq
import itertools
import matplotlib.pyplot as plt
import numpy as np

from voronoi_diagrams.src.doubly_connected_edge_list import DoublyConnectedEdgeList, PointDCEL
from voronoi_diagrams.src.events import Event, CircleEvent, make_circle_event
from voronoi_diagrams.src.math_voronoi import perpendicular_line_parameters, is_right
from voronoi_diagrams.src.point import Point2D
from voronoi_diagrams.src.trees.tree_vd import TreeVD, InternalVD, LeafVD

from typing import List, Tuple, Iterable, Sequence


class VoronoiDiagram:
    """
    Generates a 2D Voronoi diagram given a set of sites using Fortune's algorithm.

    Attributes:
        _sites (List[PointDCEL]): A set of points, each serving as a generation point for a Voronoi cell
        _queue (List[Event | CircleEvent]): Prioritizes events (sites) and circles (vertices) per the sweepline algorithm
        _tree (TreeVD): A tree for handing parabolas in Fortune's algorithm
        _dcel (DoublyConnectedEdgeList): Holds edges and vertices of the Voronoi diagram
        _n_sites (int): Number of unique sites
        _n_vertices (int): Number of vertices in the Voronoi diagram
        _n_edges (int): Number of edges in the Voronoi diagram
        _radius (float): Distance used for preventing degenerate vertices
    """

    def __init__(self, sites: Sequence[Point2D],
                 balanced_vertex_tree: bool = False,
                 radius: float = 1e-8):
        """
        Initializes the VoronoiDiagram class with a set of sites.

        Args:
            sites (Sequence[Point2D]): A set of sites for generating the Voronoi diagram
            balanced_vertex_tree (bool, optional): Determines whether the DCEL uses a balanced or unbalanced vertex tree
            radius (float, optional): Distance used for preventing degenerate point, default 1e-8
        """
        self._dcel = DoublyConnectedEdgeList(balanced_vertex_tree)
        self._tree = TreeVD()
        self._queue: List[Event | CircleEvent] = []
        self._sites = [PointDCEL(*site._x) for site in sites]
        self._n_sites = 0
        self._n_vertices = 0
        self._n_edges = 0
        self._radius = radius

        self._xmin = np.inf
        self._xmax = -np.inf
        self._ymin = np.inf
        self._ymax = -np.inf
        
    @property
    def n_sites(self) -> int:
        """
        Getter for the number of unique sites

        Return:
            int: The number of unique sites inserted
        """
        return self._n_sites

    @property
    def n_vertices(self) -> int:
        """
        Getter for the number of Voronoi vertices

        Return:
            int: The number of Voronoi vertices
        """
        return self._n_vertices

    @property
    def n_edges(self) -> int:
        """
        Getter for the number of edges in the Voronoi diagram

        Return:
            int: The number of edges
        """
        return self._n_edges

    def preprocess(self) -> None:
        """
        Preprocesses sites to handle precision issues
        """
        self._sites.sort(key=lambda p: p.y)
        i = 0
        j = 0
        n = len(self._sites)
        value = self._sites[0].y
        while True:
            while i < n and np.isclose(self._sites[i].y, value):
                i += 1
            while j < i:
                self._sites[j]._x[1] = value
                j += 1
            if i == n:
                break
            value = self._sites[i].y

        self._sites.sort(key=lambda p: p.x)
        i = 0
        j = 0
        value = self._sites[0].x
        while True:
            while i < n and np.isclose(self._sites[i].x, value):
                i += 1
            while j < i:
                self._sites[j]._x[0] = value
                j += 1
            if i == n:
                break
            value = self._sites[i].x

    def run(self) -> None:
        """
        Runs the Fortune algorithm
        """

        for site in self._sites:
            heapq.heappush(self._queue, Event(site))

        prev_site = None
        while self._queue:
            event = heapq.heappop(self._queue)
            if isinstance(event, CircleEvent):
                self._handle_circle_event(event)
            elif prev_site and prev_site == event._point:
                print("SKIPPING duplicate site", event._point)
            else:
                self._handle_site_event(event)
                prev_site = event._point
                self._n_sites += 1

    def postprocess(self, scale: float = 1.1, validate: bool = True) -> None:
        """
        Postprocess the voronoi diagram, closing any remaining edges
        along with sorting edges in the doubly connected edge list

        Args:
            scale (float, optional): Scales the size of the domain, defaults to 1.1
            validate (bool, optional): Validate the Voronoi diagram, defaults to true
        """

        if scale < 1.1:
            print('postprocess: Updating scale to min value of 1.1')
            scale = 1.1

        self._bound_voronoi_diagram(scale)

        # MUST be done after bounding to ensure all edges have been created
        # Best to do before DCEL postprocessing due to assertions
        if validate:
            euler_identity_satisfied = (self._n_vertices + 1) - self._n_edges + self._n_sites == 2
            if euler_identity_satisfied:
                print("Voronoi diagram satisfies Euler's identity")
            else:
                print("WARNING: Euler's identity is not satisfied!")
                print('   V+1 =', self._n_vertices + 1)
                print('   E =', self._n_edges)
                print('   F =', self._n_sites)
                print('   (V+1) - E + F != 2')
                print('')
                print('   Shortest edge length', self._dcel.shortest_edge_length)
                print('   Longest edge length', self._dcel.longest_edge_length)
                print('   Possible cause is the radius used for degenerate points:', self._radius)
                print('   Try adjusting the distance')
                print('')

        self._dcel.postprocess()

    def _bound_voronoi_diagram(self, scale: float) -> None:
        """
        Close all remaining edges in the Voronoi diagram

        Args:
            scale (float): Scales the size of the domain
        """

        # get and scale bounds of all vertices
        pmin, pmax = self._dcel._vertex_tree.get_bounds()
        assert pmin.dimension == 2
        assert pmax.dimension == 2
        xmin, ymin = pmin._x
        xmax, ymax = pmax._x
        # also include all sites
        xmin = min(xmin, self._xmin)
        xmax = max(xmax, self._xmax)
        ymin = min(ymin, self._ymin)
        ymax = max(ymax, self._ymax)

        dx = xmax - xmin
        xm = (xmin + xmax) / 2
        xmin = xm - scale * dx / 2
        xmax = xm + scale * dx / 2

        dy = ymax - ymin
        ym = (ymin + ymax) / 2
        ymin = ym - scale * dy / 2
        ymax = ym + scale * dy / 2
        
        def get_intersection(center: Point2D, p0: Point2D, p1: Point2D) -> PointDCEL:
            xmid = (p0.x + p1.x) / 2
            ymid = (p0.y + p1.y) / 2
            midpoint = PointDCEL(xmid, ymid)

            if np.isclose(p0.x, p1.x):
                x = xmax if p0.y > p1.y else xmin
                y = midpoint.y
            elif np.isclose(p0.y, p1.y):
                x = midpoint.x
                y = ymin if p0.x > p1.x else ymax
            else:
                a, b, c = perpendicular_line_parameters(p0, p1)
                y = ymax if p0.x < p1.x else ymin
                x = (c - b*y) / a
                if x > xmax:
                    x = xmax
                    y = (c - a*x) / b
                elif x < xmin:
                    x = xmin
                    y = (c - a*x) / b
            return PointDCEL(x, y)

        internals: Iterable[InternalVD] = \
            itertools.chain(self._tree.get_internals(), iter(self._tree._colinear_nodes))

        for node in internals:
            assert not node.edge.is_closed()
            center = node.edge.endpoints[0]
            assert center is not None
            assert isinstance(center, PointDCEL)
            intersection = get_intersection(center, *node._internal)
            assert not np.isclose(intersection.distance(center), 0)
            self._dcel.create_edge(center, intersection)
            self._n_edges += 1

    def _add_circle_event(self,
                          nodeL: LeafVD | None,
                          node: LeafVD | None,
                          nodeR: LeafVD | None) -> None:

        """
        Adds circle event to the event queue if neighboring parabolas
        in the Voronoi tree allow for it (node's segment of it parabola
        will disappear)

        Args:
            node (LeafVD | None): Node from the Voronoi tree being tested
            nodeL (LeafVD | None): Predecessor of node
            nodeR (LeafVD | None): Successor of node
        """

        if nodeL is None or node is None or nodeR is None:
            return

        # must be counterclockwise for the middle node to (eventually) disappear
        if is_right(nodeR._value, node._value, nodeL._value):
            circle = make_circle_event(nodeL._value, node._value, nodeR._value)
            assert circle is not None
            if node.circle:
                # found point within previously found circle!
                # only replace if new circle's highest point is
                # less than that of the existing circle
                # i.e. a point of the existing circle is contained
                # in the new circle
                if circle._point.y < node.circle._point.y:
                    node.circle.deactivate()
                else:
                    return

            heapq.heappush(self._queue, circle)
            node._circle = circle
            circle.node = node

    def _handle_site_event(self, event: Event) -> None:
        """
        Adds a site to the Voronoi tree along with circle events as needed

        Args:
            event (Event): Event containing a site
        """
        assert not isinstance(event, CircleEvent)

        node = self._tree.insert(event._point)

        nodeL = self._tree.get_predecessor(node)
        nodeR = self._tree.get_successor(node)
        nodeLL = None if nodeL is None else self._tree.get_predecessor(nodeL)
        nodeRR = None if nodeR is None else self._tree.get_successor(nodeR)

        self._add_circle_event(nodeLL, nodeL, node)
        self._add_circle_event(node, nodeR, nodeRR)

        self._update_min_max(event._point)

    def _handle_circle_event(self, event: CircleEvent) -> None:
        """
        Adds a vertex from the circle event to the Voronoi diagram.
        New circle events are added to the queue as needed.

        Args:
            event (CircleEvent): Event containing a vertex
        """
        if not event.is_active():
            return

        nodeL = self._tree.get_predecessor(event.node)
        nodeR = self._tree.get_successor(event.node)
        nodeLL = None if nodeL is None else self._tree.get_predecessor(nodeL)
        nodeRR = None if nodeR is None else self._tree.get_successor(nodeR)

        center = self._dcel.get_closest_vertex(event._center, radius=self._radius)
        if center is None:
            center = self._dcel.add_vertex(event._center, radius=self._radius)
            self._n_vertices += 1

        bnew, bleft, bright = self._tree.delete_node(event.node)
        bleft.edge.add_endpoint(center)
        bright.edge.add_endpoint(center)
        bnew.edge.add_endpoint(center)

        if bleft.edge.is_closed():
            new_edge = self._dcel.create_edge(*bleft.edge.endpoints)
            if new_edge is not None:
                self._n_edges += 1
        if bright.edge.is_closed():
            new_edge = self._dcel.create_edge(*bright.edge.endpoints)
            if new_edge is not None:
                self._n_edges += 1

        self._add_circle_event(nodeLL, nodeL, nodeR)
        self._add_circle_event(nodeL, nodeR, nodeRR)

    def _update_min_max(self, point):
        """
        Keeps track of bounds of the Voronoi diagram including all sites and vertices

        Args:
            point: A site or vertex in the Voronoi diagram
        """
        self._xmin = min(self._xmin, point.x)
        self._xmax = max(self._xmax, point.x)
        self._ymin = min(self._ymin, point.y)
        self._ymax = max(self._ymax, point.y)

    def plot(self, include_sites: bool = True,
             xlim: Tuple[float, float] | None = None,
             ylim: Tuple[float, float] | None = None,
             filename: str = "voronoi_diagram.png",
             show: bool = True) -> None:
        """
        Plots the Voronoi diagram

        Args:
            include_sites (bool, optional): Include dots representing the sites
            xlim (Tuple[float, float], optional): Limits for the x-axis
            ylim (Tuple[float, float], optional): Limits for the y-axis
        """

        if xlim is None:
            dx = self._xmax - self._xmin
            xlim = (self._xmin - 0.1*dx, self._xmax + 0.1*dx)

        if ylim is None:
            dy = self._ymax - self._ymin
            ylim = (self._ymin - 0.1*dy , self._ymax + 0.1*dy)

        fig = self._dcel.plot(xlim=xlim, ylim=ylim, show=False)
        ax = fig.axes[0]

        if include_sites:
            for site in self._sites:
                ax.plot(site.x, site.y, 'bo')

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if show:
            plt.show()

        fig.savefig(filename, bbox_inches='tight')
        print('Save Voronoi diagram plot to', filename)
