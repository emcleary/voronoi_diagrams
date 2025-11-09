import sys
from voronoi_diagrams.src.point import Point2D
from voronoi_diagrams.src.voronoi_diagram import VoronoiDiagram

import numpy as np


"""

When more than 3 input sites lie on the same circle, it's common to
have multiple circle events propose close but unique points thanks to
numerical errors. While this library is designed with this in mind, it
is not always possible to do handle these automatically.

It is recommended the user run the postprocessing step with validation
toggled on. When Euler's identity is not satisfied, as in this
example, it is likely that degenerate Voronoi vertices where
introduced. The user can attempt to fix this by adjusting the radius
between points, merging degenerate points within this radius.
"""


def create_points_on_circle(xc, yc, radius, n):
    points = []
    for theta in np.linspace(0, 2*np.pi, n, endpoint=False):
        x = xc + radius*np.cos(theta)
        y = yc + radius*np.sin(theta)
        points.append(Point2D(x, y))
    return points


if __name__ == '__main__':
    points = create_points_on_circle(0, 0, 100000, 100)
    lim = (-5e-8, 5e-8)

    # Degenerate example, multiple vertices near (0, 0) are inserted
    print('Example with degenerate vertices near the origin')
    degen = VoronoiDiagram(points)
    degen.preprocess()
    degen.run()
    # Include validation printouts while postprocessing, showing
    # that Euler's identity is not satisfied here.
    degen.postprocess(validate=True)
    degen.plot(show=True, xlim=lim, ylim=lim)

    print('')

    # Degenerate example, multiple vertices near (0, 0) are inserted
    print('Example with a unique vertex near the origin')
    nondegen = VoronoiDiagram(points, radius=1.9e-8)
    nondegen.preprocess()
    nondegen.run()
    # Validation will successfully pass
    nondegen.postprocess(validate=True)
    nondegen.plot(show=True, xlim=lim, ylim=lim)


