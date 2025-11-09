import pytest

from voronoi_diagrams.src.point import Point2D
from voronoi_diagrams.src.voronoi_diagram import VoronoiDiagram

from typing import List


@pytest.fixture
def voronoi_diagram(points: List[Point2D]) -> VoronoiDiagram:
    return VoronoiDiagram(points)

@pytest.fixture
def vd_preprocess(voronoi_diagram: VoronoiDiagram) -> VoronoiDiagram:
    voronoi_diagram.preprocess()
    return voronoi_diagram

@pytest.fixture
def vd_run(vd_preprocess: VoronoiDiagram) -> VoronoiDiagram:
    vd_preprocess.run()
    return vd_preprocess

@pytest.fixture
def vd_postprocess(vd_run: VoronoiDiagram) -> VoronoiDiagram:
    vd_run.postprocess()
    return vd_run
