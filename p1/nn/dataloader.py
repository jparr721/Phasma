import os
import pickle
from dataclasses import dataclass
from typing import Dict, List

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


def load_datasets(model: str, datasets_path: str) -> Dict[str, List[InputOutputGroup]]:
    folders = [
        os.path.join(datasets_path, folder) for folder in os.listdir(datasets_path)
    ]

    folders = list(
        filter(lambda x: "jelly" in x or "snow" in x or "liquid" in x, folders)
    )

    for folder in tqdm(folders):
        with open(f"{folder}.pkl", "wb+") as pf:
            pickle.dump(load_model_results(folder), pf)

    datasets = {}
    for file in os.listdir("."):
        if file.endswith(".pkl") and model in file:
            n, _ = file.split(".")
            datasets[n] = pickle.load(open(file, "rb"))
    return datasets
