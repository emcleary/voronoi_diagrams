from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

from voronoi_diagrams.complexity.timer import runtimes

from typing import List, Callable
import numpy.typing as npt


def complexity_n(x: List[float], a: float, b: float) -> npt.NDArray[np.float64]:
    return a * np.array(x, dtype=np.float64) + b

def complexity_n2(x: List[float], a: float, b: float) -> npt.NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    return a * x * x + b

def complexity_n_log_n(x: List[float], a: float, b: float) -> npt.NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    return a * x * np.log(x) + b

def complexity_log_n(x: List[float], a: float, b: float) -> npt.NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    return a * np.log(x) + b

def complexity_sqrt_n(x: List[float], a: float, b: float) -> npt.NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    return a * np.sqrt(x) + b

def best_fit(callback: Callable[[List[float], float, float], npt.NDArray[np.float64]],
             n_samples: List[int], times: List[float]) -> npt.NDArray[np.float64]:
    r = curve_fit(callback, n_samples, times)
    return callback(n_samples, *r[0])

def plot_complexity(*functions, fname=None) -> None:
    if fname is None:
        for fname in runtimes.get_function_names():
            plot_complexity(*functions, fname=fname)
        return

    x, tmed = runtimes.get_medians(fname)
    _, tmin = runtimes.get_min(fname)
    _, tmax = runtimes.get_max(fname)
    x = list(map(int, x))

    plt.plot(x, tmed)
    plt.plot(x, tmin)
    plt.plot(x, tmax)
    for function in functions:
        rv = best_fit(function, x, tmed)
        plt.plot(x, rv, '--', label=function.__name__)
    plt.ylabel('Time [s]')
    plt.xlabel('Number of sites')
    plt.legend()
    plt.title(fname)
    plt.show()
