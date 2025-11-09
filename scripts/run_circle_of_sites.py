import numpy as np
import random

from voronoi_diagrams.complexity.complexity import run_experiment
from voronoi_diagrams.src.point import Point2D

from typing import List


def create_points(n: int) -> List[Point2D]:
    points = []

    r = 1
    for theta in np.linspace(0, 2*np.pi, n, endpoint=False):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append(Point2D(x, y))

    random.shuffle(points)
    return points


if __name__ == '__main__':
    run_experiment(create_points)
