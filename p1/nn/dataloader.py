import os
import re
from collections import defaultdict

import numpy as np
from tqdm import tqdm


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

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
    sorted_folders = {os.path.basename(folder): sort_paths(folder) for folder in folders}
    for folder, paths in tqdm(sorted_folders.items()):
        for path in paths:
            folder_path = os.path.join(root, folder, path)
            files = list(
                filter(
                    lambda fn: fn == "gm.npy" or fn == "gv.npy" or fn == "x.npy",
                    os.listdir(folder_path),
                )
            )
            files = [os.path.join(folder_path, n) for n in files]
            files = {os.path.basename(os.path.dirname(n)): np.load(n) for n in files}
            loaded_datasets[folder] = files

    return loaded_datasets
