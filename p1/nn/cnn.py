import os
import sys
import threading
from dataclasses import dataclass
from time import sleep
from typing import Dict, List, Tuple

import numpy as np
import psutil
from loguru import logger
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from nn.dataloader import InputOutputGroup, load_datasets, load_pickle_files

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


def make_dataset(datasets: Dict[str, List[InputOutputGroup]]) -> Dataset:
    x = []
    y = []

    for data in tqdm(datasets.values()):
        for row in data:
            x.append(row.ig)
            y.append(row.g)

    return Dataset(np.array(x), np.array(y))


def make_model():
    inputs = Input(shape=(64, 64, 3))

    def conv_block(name, in_c, size=4, pad="same", t=False, act="relu", bn=True, do=0.3):
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

        if do > 0:
            block.add(L.Dropout(do))

        return block

    channels = int(2**6 + 0.5)
    e1 = conv_block("enc_1", 3, act="leaky_relu")(inputs)
    e2 = conv_block("enc_2", channels, act="leaky_relu")(e1)
    e3 = conv_block("enc_3", channels * 2, act="leaky_relu")(e2)
    e4 = conv_block("enc_4", channels * 4, act="leaky_relu")(e3)
    e5 = conv_block("enc_5", channels * 8, act="leaky_relu", size=2, pad="valid")(e4)
    e6 = conv_block("enc_6", channels * 8, act="leaky_relu", size=2, pad="valid")(e5)

    x = conv_block("dec_6", channels * 8, t=True, size=2, pad="valid")(e6)
    x = L.concatenate([x, e5])
    x = conv_block("dec_5", channels * 8, t=True, size=2, pad="valid")(x)
    x = L.concatenate([x, e4])
    x = conv_block("dec_4", channels * 4, t=True)(x)
    x = L.concatenate([x, e3])
    x = conv_block("dec_3", channels * 2, t=True)(x)
    x = L.concatenate([x, e2])
    x = conv_block("dec_2", channels, t=True)(x)
    x = L.concatenate([x, e1])
    x = conv_block("dec_1", 3, act=None, t=True, bn=False)(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(0.0002, beta_1=0.5, epsilon=0.1), loss="mae")
    return model


def poll_ram():
    while True:
        sleep(1)
        usage = psutil.virtual_memory().active * 1e-9

        if usage > 15:
            logger.error("Memory limit reached, killing process")
            os._exit(1)


def train_model(model):
    threading.Thread(target=poll_ram, daemon=True).start()
    base = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "datasets")

    # Check if we have cached data
    logger.info("Checking file caches")
    pkl_files = [os.path.join(base, f) for f in os.listdir(base) if f.endswith(".pkl")]

    if len(pkl_files) == 0:
        logger.warning(
            "No PKL files found, building them one by one (this can take awhile)"
        )
        pkl_files = load_datasets(base)

    while len(pkl_files) > 0:
        loaded, pkl_files = load_pickle_files(pkl_files)
        dataset = make_dataset(loaded)

        logger.success("Dataset loaded")

        x = dataset.x
        y = dataset.y

        logger.info(f"x.shape {x.shape}")
        logger.info(f"y.shape {y.shape}")

        try:
            model.fit(
                x,
                y,
                epochs=200,
                batch_size=128,
                validation_split=0.3,
                callbacks=[
                    EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
                ],
            )
        except KeyboardInterrupt:
            logger.warning("Killed. Saving state if possible.")

        # Free memory after training cycle
        del dataset
        del loaded

    return model


def save_model(model):
    dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    model.save(f"{dirname}/{_MODEL_OUTPUT_PATH}")
    logger.success("Model saved successfully")

    logger.info("Plotting")
