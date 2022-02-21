import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger
from tqdm import tqdm


@dataclass(frozen=True)
class InputOutputGroup(object):
    ig: np.ndarray
    g: np.ndarray


def load_model_result(timestep_folder_path: str):
    files = list(os.listdir(timestep_folder_path))
    ig = np.load(os.path.join(timestep_folder_path, files[files.index("ig.npy")]))
    g = np.load(os.path.join(timestep_folder_path, files[files.index("g.npy")]))
    return InputOutputGroup(ig, g)


def load_model_results(folder_path: str) -> List[InputOutputGroup]:
    timesteps = list(os.listdir(folder_path))
    groups_at_timestep = [InputOutputGroup(np.zeros([]), np.zeros([]))] * len(timesteps)
    for timestep in tqdm(timesteps):
        fullpath = os.path.join(folder_path, timestep)
        try:
            groups_at_timestep[int(timestep)] = load_model_result(fullpath)
        except Exception as e:
            logger.error(e)

    return groups_at_timestep


def load_pickle_files(
    folders: List[str], mem_limit: int = 6
) -> Tuple[Dict[str, List[InputOutputGroup]], List[str]]:
    def is_memory_at_limit(arr_arrs: List[List[InputOutputGroup]]):
        total_memory = 0
        for arrs in arr_arrs:
            for arr in arrs:
                total_memory += arr.ig.nbytes
                total_memory += arr.g.nbytes
        total_memory *= 1e-9
        return total_memory > mem_limit, total_memory

    datasets = {}
    leftovers = []
    at_memory_limit = False
    total_memory = 0
    for folder in tqdm(folders):
        n, _ = os.path.basename(folder).split(".")
        at_memory_limit, total_memory = is_memory_at_limit(list(datasets.values()))

        if not at_memory_limit:
            datasets[n] = pickle.load(open(folder, "rb"))
        else:
            leftovers.append(folder)

    logger.info(f"Using {total_memory}gb of memory.")

    return datasets, leftovers


def load_datasets(datasets_path: str) -> List[str]:
    folders = [
        os.path.join(datasets_path, folder) for folder in os.listdir(datasets_path)
    ]

    folders = list(
        filter(lambda x: "jelly" in x or "snow" in x or "liquid" in x, folders)
    )

    files = []
    for folder in tqdm(folders):
        fn = f"{folder}.pkl"
        with open(fn, "wb+") as pf:
            files.append(fn)
            pickle.dump(load_model_results(folder), pf)

    return files


@dataclass(frozen=True)
class Dataset(object):
    # Input
    x: np.ndarray

    # Targets
    y: np.ndarray
