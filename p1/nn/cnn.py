import os
import pickle
from dataclasses import dataclass
from typing import Dict, Final, List, Tuple

import numpy as np
from dataloader import InputOutputGroup, load_datasets
from loguru import logger
from tensorflow.keras import Input, Model
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tqdm import tqdm

_MODEL_OUTPUT_PATH = "cnn_model.h5"


@dataclass(frozen=True)
class Dataset(object):
    # Input
    x: np.ndarray

    # Targets
    y_F: np.ndarray
    y_advection: np.ndarray


def normalize(value: float, min_: float, max_: float) -> float:
    return (value - min_) / (max_ - min_)


def denormalize(value: float, min_: float, max_: float) -> float:
    return (value + min_) * (max_ - min_)


def _load(model: str, datasets_path: str) -> Dict[str, List[InputOutputGroup]]:
    if os.path.exists("loaded.pkl"):
        logger.success("Founded loaded dataset, using that")
        return pickle.load(open("loaded.pkl", "rb"))
    else:
        logger.warning("No cached dataset found, loading from disk")
        logger.warning("Note: if the page cache is not warm this will take _forever_")
        return load_datasets(model, datasets_path)


def make_dataset(model: str) -> Dataset:
    x = []
    y_F = []
    y_advection = []

    datasets_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "..", "datasets"
    )

    datasets = _load(model, datasets_path)

    for data in tqdm(datasets.values()):
        for row in data:
            gmgv = np.concatenate((row.gm, row.gv), axis=2)
            assert gmgv.shape == (*row.gm.shape[:2], 3), f"{gmgv.shape}"
            x.append(gmgv)
            y_F.append(row.F)
            y_advection.append(row.x)

    return Dataset(np.array(x), np.array(y_F), np.array(y_advection))


def make_model(input_shape):
    inputs = Input(shape=input_shape, name="main")

    filters: Final[int] = 32
    kernel_size: Final[int] = 5

    # Conv block 1
    x = L.Conv2D(filters=filters, kernel_size=kernel_size, padding="same")(inputs)
    x = L.LeakyReLU()(x)

    # Conv block 2
    x = L.Conv2D(filters=filters, kernel_size=kernel_size, padding="same")(x)
    x = L.LeakyReLU()(x)

    x = L.Flatten()(x)

    # Multi-Branch Output
    # Branch 1: Predict the deformation gradient.
    F_branch = L.Dense(1, activation="linear", name="F_out")(x)

    # Branch 2: Predict the advection step
    advection_branch = L.Dense(1, activation="linear", name="a_out")(x)

    return Model(inputs=inputs, outputs=[F_branch, advection_branch])


def train_model():
    dataset = make_dataset("jelly")

    logger.success("Dataset loaded")

    x = dataset.x
    y_F = dataset.y_F
    y_F_shape = y_F.shape
    y_F = y_F.reshape(y_F_shape[0], np.prod(y_F_shape[1:]))
    y_advection = dataset.y_advection
    y_advection_shape = y_advection.shape
    y_advection = y_advection.reshape(
        y_advection_shape[0], np.prod(y_advection_shape[1:])
    )

    logger.info(f"x.shape {x.shape}")
    logger.info(f"y_F.shape {y_F.shape}")
    logger.info(f"y_advection.shape {y_advection.shape}")

    model = make_model(x.shape[1:])
    model.compile(optimizer="adam", loss="mse")
    logger.info(model.summary())
    model.fit(
        {"main": x},
        {"F_out": y_F, "a_out": y_advection},
        epochs=400,
        batch_size=32,
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
