import numpy as np
import matplotlib.pyplot as plt

from src.math_voronoi import parabola_y

XMIN = -10
XMAX = 10
YMIN = -10
YMAX = 10
N = 201
x = np.linspace(XMIN, XMAX, N)

foci = [
    (-10, 1),
    (3, 6),
    (-9, 9),
    (-2, 9),
]

sweeplines = [
    9, 9.01, 9.1, 9.5, 10
]

if __name__ == '__main__':

    for i, yd in enumerate(sweeplines):
        for focus in foci:
            y = parabola_y(x, *focus, yd)
            plt.plot(*focus, 'ko')
            if y is np.inf:
                continue
            plt.plot(x, y)
        plt.plot(x, yd * np.ones(x.shape), 'k')

        plt.xlim(XMIN, XMAX)
        plt.ylim(YMIN, YMAX)
    
        plt.show()
