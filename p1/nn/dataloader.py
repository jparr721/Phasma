import os
import pickle
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from loguru import logger
from tqdm import tqdm

Group = namedtuple("Group", ["x", "gm", "gv"])


@dataclass(frozen=True)
class InputOutputGroup(object):
    x: np.ndarray
    F: np.ndarray
    gm: np.ndarray
    gv: np.ndarray


def load_model_result(timestep_folder_path: str):
    files = list(os.listdir(timestep_folder_path))
    gm = np.load(os.path.join(timestep_folder_path, files[files.index("gm.npy")]))
    gv = np.load(os.path.join(timestep_folder_path, files[files.index("gv.npy")]))
    x = np.load(os.path.join(timestep_folder_path, files[files.index("x.npy")]))
    F = np.load(os.path.join(timestep_folder_path, files[files.index("F.npy")]))
    return InputOutputGroup(x, F, gm, gv)


def load_model_results(folder_path: str) -> List[InputOutputGroup]:
    timesteps = list(os.listdir(folder_path))
    groups_at_timestep: List[InputOutputGroup] = [
        InputOutputGroup(np.zeros([]), np.zeros([]), np.zeros([]), np.zeros([]))
    ] * len(timesteps)
    for timestep in tqdm(timesteps):
        fullpath = os.path.join(folder_path, timestep)
        try:
            groups_at_timestep[int(timestep)] = load_model_result(fullpath)
        except Exception:
            logger.error(f"Folder name {timestep} is malformed")

    return groups_at_timestep


def load_datasets(model: str, datasets_path: str) -> Dict[str, List[InputOutputGroup]]:
    folders = [
        os.path.join(datasets_path, folder) for folder in os.listdir(datasets_path)
    ]

    folders = list(filter(lambda x: model in x, folders))

    datasets = {
        os.path.basename(folder): load_model_results(folder) for folder in tqdm(folders)
    }

    with open("loaded.pkl", "wb+") as pf:
        pickle.dump(datasets, pf)

    return datasets
