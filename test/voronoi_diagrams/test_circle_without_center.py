import numpy as np
import pytest

from voronoi_diagrams.src.doubly_connected_edge_list import PointDCEL
from voronoi_diagrams.src.math_voronoi import is_left
from voronoi_diagrams.src.point import Point2D
from voronoi_diagrams.src.voronoi_diagram import VoronoiDiagram
from voronoi_diagram_fixtures import *

from typing import List


# Generates sites on a unit circle
def generate_sites(n: int) -> List[Point2D]:
    points = []
    for theta in np.linspace(0, 2*np.pi, n, endpoint=False):
        x = np.cos(theta)
        y = np.sin(theta)
        points.append(Point2D(x, y))
    return points

@pytest.fixture(params=[3, 4, 5, 6])
def n_sites(request) -> int:
    return request.param

@pytest.fixture
def points(n_sites: int) -> List[Point2D]:
    return generate_sites(n_sites)

@pytest.fixture
def origin(vd_postprocess: VoronoiDiagram) -> PointDCEL | None:
    return vd_postprocess._dcel.get_closest_vertex(Point2D(0, 0))

def test_constructor(voronoi_diagram: VoronoiDiagram, n_sites: int) -> None:
    assert len(voronoi_diagram._sites) == n_sites
    assert len(voronoi_diagram._dcel._vertices) == 0

def test_preprocess(vd_preprocess: VoronoiDiagram) -> None:
    assert len(vd_preprocess._dcel._vertices) == 0
    assert len(vd_preprocess._dcel._edges) == 0

def test_run(vd_run: VoronoiDiagram) -> None:
    # only 1 vertex given circle of sites
    # test to ensure no degeneracy
    assert len(vd_run._dcel._vertices) == 1

    # expect no edges to be closed yet
    assert len(vd_run._dcel._edges) == 0
    
    # expect vertex to be at origin given the set of points
    vertex = vd_run._dcel._vertices[0]
    assert np.isclose(vertex.x, 0, atol=1e-15)
    assert np.isclose(vertex.y, 0, atol=1e-15)
    
def test_postprocess(vd_postprocess: VoronoiDiagram, n_sites: int) -> None:
    # expect 1 vertex for the origin only
    assert len(vd_postprocess._dcel._vertices) == 1
    # expect 2 (half)edges for each site
    assert len(vd_postprocess._dcel._edges) == 2 * n_sites

def test_origin(origin: PointDCEL | None) -> None:
    assert origin is not None
    assert np.isclose(origin.x, 0, atol=1e-15)
    assert np.isclose(origin.y, 0, atol=1e-15)

def test_dcel(origin: PointDCEL | None) -> None:
    assert origin is not None
    # confirms the dcel is postprocessed correctly on an open mesh
    # 1) expect next == twin
    # 2) expect prev.prev == counterclockwise rotation about the origin
    edge = origin.edge
    assert edge is not None
    while True:
        assert edge.twin is edge.next
        assert edge.next.twin is edge

        # confirm boundary points are in the counterclockwise direction
        edge_rot = edge.prev.prev
        # not ALWAYS true for any VD, but always true for sites on a circle
        assert is_left(origin, edge.dest, edge_rot.dest)

        edge = edge_rot
        if edge is origin.edge:
            break


if __name__ == '__main__':
    n = 3
    pts = generate_sites(n)
    vd = VoronoiDiagram(pts)
    vd.preprocess()
    vd.run()
    vd.postprocess()
    vd.plot()

    def get_bounds():
        xmin = np.inf
        xmax = -np.inf
        ymin = np.inf
        ymax = -np.inf
        for point in pts:
            xmin = min(xmin, point.x)
            xmax = max(xmax, point.x)
            ymin = min(ymin, point.y)
            ymax = max(ymax, point.y)
        return xmin, xmax, ymin, ymax
    b = get_bounds()
    point_origin = vd._dcel.get_closest_vertex(Point2D(0, 0))
    test_origin(point_origin)
    test_dcel(point_origin)
