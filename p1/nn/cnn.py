import os
import threading
from time import sleep
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from loguru import logger
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam

from nn.dataloader import Dataset, InputOutputGroup, load_datasets, load_pickle_files

_MODEL_OUTPUT_PATH = "cnn_model.h5"
_BATCH_SIZE = 20
_EPOCHS = 50
_LR = 0.001
_LOSS = "mae"
_CHANNELS = 3


def make_dataset(datasets: Dict[str, List[InputOutputGroup]]) -> Dataset:
    return Dataset(
        np.stack([v.x for ls in datasets.values() for v in ls]),
        np.stack([v.y for ls in datasets.values() for v in ls]),
    )


def make_model(*, input_shape=(64, 64, _CHANNELS), expo=6, dropout=0.0):
    def conv_block(
        name, filters, kernel_size=4, pad="same", t=False, act="relu", bn=True
    ):
        block = Sequential(name=name)

        if act == "relu":
            block.add(L.ReLU())
        elif act == "leaky_relu":
            block.add(L.LeakyReLU(0.2))

        if not t:
            block.add(
                L.Conv2D(
                    filters,
                    kernel_size=kernel_size,
                    strides=(2, 2),
                    padding=pad,
                    use_bias=True,
                    activation=None,
                    kernel_initializer=RandomNormal(0.0, 0.2),
                )
            )
        else:
            block.add(L.UpSampling2D(interpolation="bilinear"))
            block.add(
                L.Conv2DTranspose(
                    filters=filters,
                    kernel_size=kernel_size - 1,
                    padding=pad,
                    activation=None,
                )
            )

        if dropout > 0:
            block.add(L.SpatialDropout2D(dropout))

        if bn:
            block.add(L.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9))

        return block

    channels = int(2**expo + 0.5)
    e0 = Sequential(name="enc_0")
    e0.add(
        L.Conv2D(
            filters=_CHANNELS,
            kernel_size=4,
            strides=(2, 2),
            padding="same",
            activation=None,
            data_format="channels_last",
        )
    )
    e1 = conv_block("enc_1", channels, act="leaky_relu")
    e2 = conv_block("enc_2", channels * 2, act="leaky_relu")
    e3 = conv_block("enc_3", channels * 4, act="leaky_relu")
    e4 = conv_block("enc_4", channels * 8, act="leaky_relu", kernel_size=2, pad="valid")
    e5 = conv_block("enc_5", channels * 8, act="leaky_relu", kernel_size=2, pad="valid")

    dec_5 = conv_block("dec_5", channels * 8, t=True, kernel_size=2, pad="valid")
    dec_4 = conv_block("dec_4", channels * 8, t=True, kernel_size=2, pad="valid")
    dec_3 = conv_block("dec_3", channels * 4, t=True)
    dec_2 = conv_block("dec_2", channels * 2, t=True)
    dec_1 = conv_block("dec_1", channels, act=None, t=True, bn=False)
    dec_0 = Sequential(name="dec_0")
    dec_0.add(L.ReLU())
    dec_0.add(L.Conv2DTranspose(_CHANNELS, kernel_size=4, strides=(2, 2), padding="same"))

    # Forward Pass
    inputs = Input(shape=input_shape)
    out0 = e0(inputs)
    out1 = e1(out0)
    out2 = e2(out1)
    out3 = e3(out2)
    out4 = e4(out3)
    out5 = e5(out4)

    dout5 = dec_5(out5)
    dout5_out4 = tf.concat([dout5, out4], axis=3)
    dout4 = dec_4(dout5_out4)
    dout4_out3 = tf.concat([dout4, out3], axis=3)
    dout3 = dec_3(dout4_out3)
    dout3_out2 = tf.concat([dout3, out2], axis=3)
    dout2 = dec_2(dout3_out2)
    dout2_out1 = tf.concat([dout2, out1], axis=3)
    dout1 = dec_1(dout2_out1)
    dout1_out0 = tf.concat([dout1, out0], axis=3)
    dout0 = dec_0(dout1_out0)

    model = Model(inputs=inputs, outputs=dout0)
    model.compile(optimizer=Adam(_LR, beta_1=0.5), loss=_LOSS)
    print(model.summary())
    return model


def poll_ram():
    while True:
        sleep(1)

        try:
            used = int(
                os.popen("free --giga --total | awk '/^Total:/ {print $3}'").read()
            )

            if used > 31:
                logger.error("Memory limit reached, killing process")
                logger.warning("Memory will not be cleaned up, I suggest a reboot")
                os._exit(1)
        except Exception as e:
            logger.error(f"Error getting memory reading! This is dangerous! {e}")


def train_model(model: Model, show_plots=False):
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

    x_scaler = StandardScaler()
    while len(pkl_files) > 0:
        loaded, pkl_files = load_pickle_files(pkl_files, 5)
        dataset = make_dataset(loaded)

        logger.success("Dataset loaded")

        # x = x_scaler.fit_transform(dataset.x.reshape(-1, dataset.x.shape[-1])).reshape(
        #     dataset.x.shape
        # )
        x = np.moveaxis(dataset.x, 1, -1)
        y = np.moveaxis(dataset.y, 1, -1)

        logger.info(f"x.shape {x.shape}")
        logger.info(f"y.shape {y.shape}")

        try:
            history = model.fit(
                x,
                y,
                epochs=_EPOCHS,
                batch_size=_BATCH_SIZE,
                validation_split=0.3,
                callbacks=[
                    EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
                ],
            )
            if show_plots:
                plot(history)
        except Exception as e:
            raise e

        # Free memory after training cycle
        del dataset
        del loaded

    return model


def plot(history: History):
    logger.info("Plotting")
    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]

    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training & Validation Loss", fontsize=16)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend()

    plt.show()


def save_model(model):
    dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    model.save(f"{dirname}/{_MODEL_OUTPUT_PATH}")
    logger.success("Model saved successfully")

    logger.info("Plotting")
