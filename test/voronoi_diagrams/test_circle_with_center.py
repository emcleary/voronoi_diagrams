import numpy as np
import pytest

from voronoi_diagrams.src.doubly_connected_edge_list import PointDCEL
from voronoi_diagrams.src.point import Point2D
from voronoi_diagrams.src.voronoi_diagram import VoronoiDiagram
from voronoi_diagram_fixtures import *

from typing import List


# Generate sites on a unit circle plus the origin
def generate_sites(n: int) -> List[Point2D]:
    points: List[Point2D] = []
    for theta in np.linspace(0, 2*np.pi, n, endpoint=False):
        x = np.cos(theta)
        y = np.sin(theta)
        points.append(Point2D(x, y))
    points.append(Point2D(0, 0))
    return points

@pytest.fixture(params=[3, 4, 5, 6])
def n_sites(request) -> int:
    return request.param

@pytest.fixture
def points(n_sites: int) -> List[Point2D]:
    return generate_sites(n_sites)

@pytest.fixture
def vd_postprocess(vd_run: VoronoiDiagram) -> VoronoiDiagram:
    vd_run.postprocess()
    return vd_run

@pytest.fixture
def origin(vd_postprocess: VoronoiDiagram) -> PointDCEL | None:
    return vd_postprocess._dcel.get_closest_vertex(Point2D(0, 0))

def test_constructor(voronoi_diagram: VoronoiDiagram, n_sites: int) -> None:
    assert len(voronoi_diagram._sites) == 1 + n_sites
    assert len(voronoi_diagram._dcel._vertices) == 0
    assert len(voronoi_diagram._dcel._edges) == 0

def test_preprocess(vd_preprocess: VoronoiDiagram) -> None:
    assert len(vd_preprocess._dcel._vertices) == 0
    assert len(vd_preprocess._dcel._edges) == 0

def test_run(vd_run: VoronoiDiagram, n_sites: int) -> None:
    assert len(vd_run._dcel._vertices) == n_sites
    assert len(vd_run._dcel._edges) == 2 * n_sites
    
def test_postprocess(vd_postprocess: VoronoiDiagram, n_sites: int) -> None:
    assert len(vd_postprocess._dcel._vertices) == n_sites
    assert len(vd_postprocess._dcel._edges) == 4 * n_sites

def test_origin(origin: PointDCEL) -> None:
    assert origin is None

if __name__ == '__main__':
    n = 3
    pts = generate_sites(n)
    vd = VoronoiDiagram(pts)
    vd.preprocess()
    vd.run()
    vd.postprocess()
    vd.plot()
