import os
import re
from collections import defaultdict

import numpy as np


def sort_paths(folder: str):
    return sorted(
        list(os.listdir(folder)),
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split("(\\d+)", s)],
    )


def load_datasets(datasets_path: str):
    loaded_datasets = defaultdict(dict)
    folders = [
        os.path.join(datasets_path, folder) for folder in os.listdir(datasets_path)
    ]

    sorted_folders = defaultdict(list)
    for folder in folders:
        sorted_folders[os.path.basename(folder)] = sort_paths(folder)
        sorted_folders[os.path.basename(folder)] = [
            os.path.join(folder, subfolder)
            for subfolder in sorted_folders[os.path.basename(folder)]
        ]

    for i, folder_name in enumerate(sorted_folders.keys()):
        paths = sorted_folders[folder_name]
        for path in paths:
            if os.path.basename(path).lower() == "gm.npy":
                loaded_datasets[folder_name][i] = np.load(path)
            if os.path.basename(path).lower() == "gv.npy":
                loaded_datasets[folder_name][i] = np.load(path)
            if os.path.basename(path).lower() == "x.npy":
                loaded_datasets[folder_name][i] = np.load(path)

    return loaded_datasets
