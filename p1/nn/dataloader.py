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
    igm: np.ndarray
    igv: np.ndarray
    gm: np.ndarray
    gv: np.ndarray


def load_model_result(timestep_folder_path: str):
    files = list(os.listdir(timestep_folder_path))
    igm = np.load(os.path.join(timestep_folder_path, files[files.index("igm.npy")]))
    igv = np.load(os.path.join(timestep_folder_path, files[files.index("igv.npy")]))
    gm = np.load(os.path.join(timestep_folder_path, files[files.index("gm.npy")]))
    gv = np.load(os.path.join(timestep_folder_path, files[files.index("gv.npy")]))
    return InputOutputGroup(igm, igv, gm, gv)


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
