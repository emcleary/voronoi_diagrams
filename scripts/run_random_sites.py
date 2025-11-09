import numpy as np

from voronoi_diagrams.complexity.complexity import run_experiment
from voronoi_diagrams.src.point import Point2D


XMIN = -10000
XMAX = 10000
YMIN = -10000
YMAX = 10000


def create_points(n: int) -> List[Point2D]:
    points = []

    for _ in range(n):
        x = np.random.uniform(XMIN, XMAX, 1)[0]
        y = np.random.uniform(YMIN, YMAX, 1)[0]
        points.append(Point2D(x, y))

    return points


if __name__ == '__main__':
    run_experiment(create_points)
