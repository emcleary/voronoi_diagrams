from argparse import ArgumentParser
from pathlib import Path
import numpy as np

from voronoi_diagrams.complexity.performance import *
from voronoi_diagrams.complexity.timer import timer, runtimes, dump_runtimes, load_runtimes
from voronoi_diagrams.src.point import Point2D
from voronoi_diagrams.src.voronoi_diagram import VoronoiDiagram

from typing import List, Callable


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('samples', type=int, help="Number of samples per set of sites")
    parser.add_argument('n_sites', type=int, nargs='+', help="Number of sites")
    parser.add_argument('--outfile', type=str, default='result.json', help="Output filename")
    parser.add_argument('--balanced', action='store_true')
    args = parser.parse_args()
    if not args.outfile.endswith('.json'):
        print("Outfile must end with '.json'")
        exit()
    args.n_sites.sort()
    return args

@timer
def execute_all(vd):
    preprocess(vd)
    run(vd)
    postprocess(vd)

@timer
def preprocess(vd: VoronoiDiagram) -> None:
    vd.preprocess()

@timer
def run(vd: VoronoiDiagram) -> None:
    vd.run()

@timer
def postprocess(vd: VoronoiDiagram) -> None:
    vd.postprocess(scale=2, validate=False)

def run_experiment(create_points: Callable[[int], List[Point2D]]):
    args = parse_arguments()
    
    datafile = Path(args.outfile)
    if datafile.is_file():
        load_runtimes(datafile)

    calc_n = lambda x : 1 << x
    balanced_vertex_tree = False
    for n in args.n_sites:
        print('Number of points:', n, 'of', args.n_sites[-1])
        for _ in range(args.samples):
            seed = np.random.randint(0, 1<<31 - 1)
            np.random.seed(seed)
            points = create_points(n)
            try:
                vd = VoronoiDiagram(points, args.balanced)
                execute_all(vd)
            except:
                print("FAILED ON SEED", seed, 'with', n, 'points')
        runtimes.store(n)
        dump_runtimes(datafile)
