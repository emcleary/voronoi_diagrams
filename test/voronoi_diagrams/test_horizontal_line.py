import numpy as np
import pytest

from voronoi_diagrams.src.voronoi_diagram import VoronoiDiagram
from voronoi_diagrams.src.point import Point2D
from voronoi_diagram_fixtures import *

from typing import List


# Generate sites on a horizontal line,
# plus one point above the line
def generate_sites(n: int) -> List[Point2D]:
    points = []
    y = -1.0
    xmin = -1.0
    xmax = 1.0
    for x in np.linspace(xmin, xmax, n, endpoint=True):
        points.append(Point2D(x, y))
    points.append(Point2D(0.0, 0.0))
    return points

@pytest.fixture
def points(n_sites: int) -> List[Point2D]:
    return generate_sites(n_sites)

@pytest.fixture(params=[2, 3, 4, 5, 10, 15])
def n_sites(request) -> int:
    return request.param

@pytest.fixture
def vd_postprocess(vd_run: VoronoiDiagram) -> VoronoiDiagram:
    vd_run.postprocess()
    return vd_run

def test_constructor(voronoi_diagram: VoronoiDiagram, n_sites: int) -> None:
    assert len(voronoi_diagram._sites) == 1 + n_sites
    assert len(voronoi_diagram._dcel._vertices) == 0
    assert len(voronoi_diagram._dcel._edges) == 0

def test_preprocess(vd_preprocess: VoronoiDiagram) -> None:
    assert len(vd_preprocess._dcel._vertices) == 0
    assert len(vd_preprocess._dcel._edges) == 0

def test_run(vd_run: VoronoiDiagram, n_sites: int) -> None:
    assert len(vd_run._dcel._vertices) == n_sites - 1
    assert len(vd_run._dcel._edges) == 2 * (n_sites - 2)
    
def test_postprocess(vd_postprocess: VoronoiDiagram, n_sites: int) -> None:
    assert len(vd_postprocess._dcel._vertices) == n_sites - 1
    assert len(vd_postprocess._dcel._edges) == 2 * (2*n_sites - 1)


if __name__ == '__main__':
    n = 5
    pts = generate_sites(n)
    vd = VoronoiDiagram(pts)
    vd.preprocess()
    vd.run()
    print(vd._dcel._edges)
    vd.postprocess()
    vd.plot()
