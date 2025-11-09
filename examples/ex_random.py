from voronoi_diagrams.src.point import Point2D
from voronoi_diagrams.src.voronoi_diagram import VoronoiDiagram

import numpy as np
from argparse import ArgumentParser

"""
This example runs the library using a random set of sites.
These sites are sampled from uniform distributions for both
x- and y-coordinates.
"""

def parse_arguments():
    parser = ArgumentParser(
        prog='Example Random',
        description='Example Voronoi diagram using a random set of sites'
    )
    parser.add_argument('xmin', type=float, help='Lower bound for x')
    parser.add_argument('xmax', type=float, help='Upper bound for x')
    parser.add_argument('ymin', type=float, help='Lower bound for y')
    parser.add_argument('ymax', type=float, help='Upper bound for y')
    parser.add_argument('n', type=int, help='Number of sites')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed (randomly selected for a negative input)')
    args = parser.parse_args()
    if args.seed < 0:
        args.seed = np.random.randint(0, 1<<31 - 1)
        print('Random seed set to', args.seed)
    return args


def create_random_points(xmin, xmax, ymin, ymax, n):
    points = []
    for _ in range(n):
        x = np.random.uniform(xmin, xmax, 1)[0]
        y = np.random.uniform(ymin, ymax, 1)[0]
        points.append(Point2D(x, y))
    return points


if __name__ == '__main__':
    args = parse_arguments()
    np.random.seed(args.seed)

    points = create_random_points(args.xmin, args.xmax,
                                  args.ymin, args.ymax, args.n)

    vd = VoronoiDiagram(points)
    vd.preprocess()
    vd.run()
    vd.postprocess(scale=3)
    vd.plot(show=True)
