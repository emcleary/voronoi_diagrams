from pathlib import Path
import json
import numpy as np
import time

from numpy.typing import NDArray
from typing import Dict, List, Callable, Tuple, Iterable

RT = Dict[str, List[float]]

class Runtimes:
    """
    Tracks runtimes of methods and functions

    Attributes:
        _data (Dict[str, RT]): Data for function names, number of sites and runtimes
        _current (RT): Data for function names and runtimes
    """
    
    def __init__(self):
        """
        Constructs a Runtimes object
        """
        self._data: Dict[str, RT] = dict()
        self._current: RT = dict()

    def has_any(self) -> bool:
        """
        Checks if any runtime data has been stored

        Return:
            bool: True if any runtime data has been stored, false otherwise
        """
        return len(self._data) > 0

    def has_any_not_set(self) -> bool:
        """
        Checks if any runtime data has not been set to a number of sites

        Return:
            bool: True if any runtime data has been gathered but not stored, false otherwise
        """
        return len(self._current) > 0

    def append(self, fname: str, value: float) -> None:
        """
        Store a datapoint

        Args:
            fname (str): Function name
            value (float): A runtime measurement
        """
        if fname not in self._current:
            self._current[fname] = []
        self._current[fname].append(value)

    def store(self, n: int | str) -> None:
        """
        Store current data corresponding to a size n

        Args:
            n (int | str): Size (e.g. number of sites)
        """
        # string typing for consistency with json files
        if isinstance(n, int):
            n = str(n)
        if self._current:
            for fname, times in self._current.items():
                if fname not in self._data:
                    self._data[fname] = dict()
                if n not in self._data[fname]:
                    self._data[fname][n] = []
                self._data[fname][n] += times
            self._current = dict()

    def get_data(self, callback: Callable[[Iterable[float]], float]
                 ) -> Dict[str, Tuple[NDArray[np.int_], NDArray[np.floating]]]:
        """
        Filter data using some callback

        Return:
   
            Dict[str, Tuple[npt.NDArray[int], npt.NDArray[float]]]: Filtered data
                with size and time for each function name
        """
        data = dict()
        for fname, values in self._data.items():
            d = [(int(n), callback(t)) for n, t in values.items()]
            d.sort()
            n_sites, times = zip(*d)
            n_sites = np.asarray(n_sites, dtype=int)
            times = np.asarray(times, dtype=float)
            data[fname] = (n_sites, times)
        return data

    def get_min(self) -> Dict[str, Tuple[NDArray[np.int_], NDArray[np.floating]]]:
        """
        Filters data to get minimum runtimes for each function and size

        Return:
            Dict[str, Tuple[npt.NDArray[int], npt.NDArray[float]]]: Filtered data
                with size and minimum time for each function name
        """
        return self.get_data(min)

    def get_medians(self) -> Dict[str, Tuple[NDArray[np.int_], NDArray[np.floating]]]:
        """
        Filters data to get median runtimes for each function and size

        Return:
            Dict[str, Tuple[npt.NDArray[int], npt.NDArray[float]]]: Filtered data
                with size and median time for each function name
        """

        return self.get_data(np.median) # type: ignore

    def get_max(self) -> Dict[str, Tuple[NDArray[np.int_], NDArray[np.floating]]]:
        """
        Filters data to get maximum runtimes for each function and size

        Return:
            Dict[str, Tuple[npt.NDArray[int], npt.NDArray[float]]]: Filtered data
                with size and maximum time for each function name
        """
        return self.get_data(max)

    def dump(self, filename: Path) -> None:
        """
        Write data to a file

        Args:
            filename (Path): Name of file for writing data
        """
        
        print('Dumping runtimes to', filename)
        with open(filename, 'w') as file:
            json.dump(self._data, file, indent=2)

    def load(self, filename: Path) -> None:
        """
        Load data from a file

        Args:
            filename (Path): Name of file for reading data
        """
        with open(filename, 'r') as file:
            data = json.load(file)

        assert len(self._current) == 0
        for function, values in data.items():
            for n, times in values.items():
                for t in times:
                    self.append(function, t)
                self.store(n)
                


runtimes = Runtimes()


def timer(f):
    def function(*args, **kwargs):
        tic = time.time()
        result = f(*args, **kwargs)
        toc = time.time()
        runtimes.append(f.__name__, toc - tic)
        return result
    return function


def load_runtimes(filename: Path) -> None:
    runtimes.load(filename)
        

def dump_runtimes(filename: Path) -> None:
    runtimes.dump(filename)
        

def display_runtimes():
    if runtimes.has_any_not_set():
        if not runtimes.has_any():
            runtimes.store(-1)
        else:
            print('Skipping unset runtime data')

    if not runtimes.has_any():
        return

    print('')
    print('Function runtimes:')
    print('')
    data = runtimes.get_max()
    for fname, (n_sites, times) in data.items():
        print(f'{fname}:')
        for n, t in zip(n_sites, times):
            print(f'   {int(n)} {t}')

import atexit
atexit.register(display_runtimes)
