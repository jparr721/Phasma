import os
import pickle
from dataclasses import dataclass
from typing import Dict, Final, List

import numpy as np
from dataloader import InputOutputGroup, load_datasets
from loguru import logger
from tensorflow.keras import Input, Model
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm

_MODEL_OUTPUT_PATH = "cnn_model.h5"


@dataclass(frozen=True)
class Dataset(object):
    # Input
    x: np.ndarray

    # Targets
    y: np.ndarray


def normalize(value: float, min_: float, max_: float) -> float:
    return (value - min_) / (max_ - min_)


def denormalize(value: float, min_: float, max_: float) -> float:
    return (value + min_) * (max_ - min_)


def _load(model: str, datasets_path: str) -> Dict[str, List[InputOutputGroup]]:
    logger.info("Checking file caches")
    datasets = {}
    for _file in tqdm(os.listdir(datasets_path)):
        file = os.path.join(datasets_path, _file)
        if file.endswith(".pkl") and model in file:
            n, _ = _file.split(".")
            datasets[n] = pickle.load(open(file, "rb"))

    if len(datasets) > 0:
        logger.success("Cached dataset found and loaded")
        return datasets
    else:
        logger.warning("No cached dataset found, loading from disk")
        logger.warning("Note: if the page cache is not warm this will take _forever_")
        return load_datasets(model, datasets_path)


def make_dataset(model: str) -> Dataset:
    x = []
    y = []

    datasets_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "..", "datasets"
    )

    datasets = _load(model, datasets_path)

    for data in tqdm(datasets.values()):
        for row in data:
            igmigv = np.concatenate((row.igm, row.igv), axis=2)
            assert igmigv.shape == (*row.igm.shape[:2], 3), f"{igmigv.shape}"
            x.append(igmigv)

            gmgv = np.concatenate((row.gm, row.gv), axis=2)
            assert gmgv.shape == (*row.gm.shape[:2], 3), f"{gmgv.shape}"
            y.append(gmgv)

    return Dataset(np.array(x), np.array(y))


def make_model(input_shape):
    inputs = Input(shape=input_shape)

    filters: Final[int] = 32
    kernel_size: Final[int] = 5

    def conv_block(x, f, ks):
        x = L.Conv2D(filters=f, kernel_size=ks, padding="same")(x)
        x = L.LeakyReLU()(x)
        return x

    # Conv block 1
    block_1 = conv_block(inputs, filters, kernel_size)

    # Conv block 2
    block_2 = conv_block(block_1, filters, 1)

    x = L.Conv2D(filters=filters, kernel_size=kernel_size, padding="same")(block_2)

    # Skip block
    skip_block_1 = L.add([block_1, x])

    # Block 3
    outputs = L.LeakyReLU()(skip_block_1)
    outputs = L.BatchNormalization()(outputs)

    return Model(inputs=inputs, outputs=outputs)


def train_model():
    dataset = make_dataset("jelly")

    logger.success("Dataset loaded")

    x = dataset.x
    y = dataset.y

    logger.info(f"x.shape {x.shape}")
    logger.info(f"y.shape {y.shape}")

    model = make_model(x.shape[1:])
    model.compile(optimizer="adam", loss="mse")
    logger.info(model.summary())
    model.fit(
        x,
        y,
        epochs=400,
        batch_size=8,
        validation_split=0.3,
        callbacks=[EarlyStopping(monitor="loss", patience=20, restore_best_weights=True)],
    )
    return model


def save_model(model):
    dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    model.save(f"{dirname}/{_MODEL_OUTPUT_PATH}")
    logger.success("Model saved successfully")

    logger.info("Plotting")


if __name__ == "__main__":
    save_model(train_model())
