import os
import pickle
import sys
from dataclasses import dataclass
from typing import Dict, Final, List

import numpy as np
from dataloader import InputOutputGroup, load_datasets
from loguru import logger
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.saving.save import load_model
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
            x.append(row.ig)
            y.append(row.g)

    return Dataset(np.array(x), np.array(y))


def make_model():
    inputs = Input(shape=(64, 64, 3))

    def conv_block(name, in_c, size=4, pad="same", t=False, act="relu", bn=True):
        block = Sequential(name=name)

        if not t:
            block.add(
                L.Conv2D(
                    in_c,
                    kernel_size=size,
                    strides=2,
                    padding=pad,
                    use_bias=True,
                    activation=act,
                )
            )
        else:
            block.add(L.UpSampling2D(interpolation="bilinear"))
            block.add(
                L.Conv2DTranspose(
                    filters=in_c,
                    kernel_size=size - 1,
                    padding=pad,
                    activation=act,
                )
            )

        if bn:
            block.add(L.BatchNormalization())
        return block

    channels = int(2**6 + 0.5)
    e1 = conv_block("enc_1", 3, act="leaky_relu")(inputs)
    e2 = conv_block("enc_2", channels, act="leaky_relu")(e1)
    e3 = conv_block("enc_3", channels * 2, act="leaky_relu")(e2)
    e4 = conv_block("enc_4", channels * 2, act="leaky_relu")(e3)
    e5 = conv_block("enc_5", channels * 4, act="leaky_relu", size=2, pad="valid")(e4)
    e6 = conv_block("enc_6", channels * 4, act="leaky_relu", size=2, pad="valid")(e5)

    x = conv_block("dec_6", channels * 4, t=True, size=2, pad="valid")(e6)
    x = L.concatenate([x, e5])
    x = conv_block("dec_5", channels * 4, t=True, size=2, pad="valid")(x)
    x = L.concatenate([x, e4])
    x = conv_block("dec_4", channels * 2, t=True)(x)
    x = L.concatenate([x, e3])
    x = conv_block("dec_3", channels * 2, t=True)(x)
    x = L.concatenate([x, e2])
    x = conv_block("dec_2", channels, t=True)(x)
    x = L.concatenate([x, e1])
    x = conv_block("dec_1", 3, act=None, t=True, bn=False)(x)

    return Model(inputs=inputs, outputs=x)


def train_model(model, sim):
    dataset = make_dataset(sim)

    logger.success("Dataset loaded")

    x = dataset.x
    y = dataset.y

    logger.info(f"x.shape {x.shape}")
    logger.info(f"y.shape {y.shape}")

    try:
        model.fit(
            x,
            y,
            epochs=50,
            batch_size=128,
            validation_split=0.3,
            callbacks=[
                EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
            ],
        )
    except Exception:
        pass

    return model


def save_model(model):
    dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    model.save(f"{dirname}/{_MODEL_OUTPUT_PATH}")
    logger.success("Model saved successfully")

    logger.info("Plotting")


if __name__ == "__main__":
    sim = sys.argv[1]
    logger.info(f"Running {sim}")

    if sim == "jelly":
        model = make_model()
        model.compile(optimizer=Adam(0.0002, beta_1=0.5), loss="mse")
        logger.info(model.summary())
    else:
        dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        model = load_model(f"{dirname}/{_MODEL_OUTPUT_PATH}")

    save_model(train_model(model, sim))
