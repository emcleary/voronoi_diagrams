from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from voronoi_diagrams.complexity.performance import best_fit, complexity_n_log_n
from voronoi_diagrams.complexity.timer import runtimes, load_runtimes

from typing import List, Callable


def plot(n_sites, times, title, outfile):
    v = best_fit(complexity_n_log_n, n_sites, times)
    v -= 10
    n_sites = n_sites / 10000

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(n_sites, times, '-o', label='runtimes')
    ax.plot(n_sites, v, '--', label='n log(n)')
    
    ax.set_xlabel('Number of sites / 10,000')
    ax.set_ylabel('Time [s]')
    ax.legend()
    ax.set_title(title)

    fig.savefig(outfile)
    
    return fig, ax


if __name__ == '__main__':
    datafile = Path('scripts/performance_random.json')
    assert datafile.is_file()
    load_runtimes(datafile)

    function_name = 'execute_all'

    plot(*runtimes.get_medians()[function_name], "Median Rutime", "complexity_random_median.png")
    plot(*runtimes.get_max()[function_name], "Maximum Rutime", "complexity_random_max.png")
    plot(*runtimes.get_min()[function_name], "Minimum Rutime", "complexity_random_min.png")
