from voronoi_diagrams.src.point import Point2D
from voronoi_diagrams.src.voronoi_diagram import VoronoiDiagram

import numpy as np
from argparse import ArgumentParser

"""
This example runs the library using a set of sites on a circle.
"""

def parse_arguments():
    parser = ArgumentParser(
        prog='Example Circle',
        description='Example Voronoi diagram using a set of sites on a circle'
    )        
    parser.add_argument('-x', type=float, default=0, help='x-coordinate of circle center')
    parser.add_argument('-y', type=float, default=0, help='y-coordinate of circle center')
    parser.add_argument('-r', '--radius', type=float, default=1, help='Radius of circle')
    parser.add_argument('n', type=int, help='Number of sites')
    return parser.parse_args()


def create_points_on_circle(xc, yc, radius, n):
    points = []
    for theta in np.linspace(0, 2*np.pi, n, endpoint=False):
        x = xc + radius*np.cos(theta)
        y = yc + radius*np.sin(theta)
        points.append(Point2D(x, y))
    return points


if __name__ == '__main__':
    args = parse_arguments()
    points = create_points_on_circle(args.x, args.y, args.radius, args.n)

    vd = VoronoiDiagram(points)
    vd.preprocess()
    vd.run()
    vd.postprocess(scale=3)
    vd.plot(show=True)
