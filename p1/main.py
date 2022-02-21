import os
from typing import Final, List

import taichi as ti
import typer
from loguru import logger
from tensorflow.keras.models import load_model
from tqdm import tqdm

from mpm.engine import Engine, cube
from mpm.mpm import *
from nn.cnn import make_model, save_model, train_model

app = typer.Typer(help="p1")


@app.command()
def offline_sim(
    saved: List[str] = typer.Option([]),
    outdir: str = typer.Option("tmp"),
    use_gui: bool = typer.Option(False),
):
    bounds = (0.4, 0.6)
    shape_res = 25
    c1 = cube(bounds, shape_res)
    c1[:, 1] -= 0.35
    c2 = cube(bounds, shape_res)

    x = np.concatenate((c1, c2))
    e = Engine(outdir)
    e.simulate(x, use_gui=use_gui, saved=saved)


@app.command()
def gen_data(
    model: str = typer.Option("jelly"),
    steps: int = typer.Option(4000),
    gmin: int = typer.Option(-10),
    gmax: int = typer.Option(-300),
    incr: int = typer.Option(10),
):
    e = Engine()

    bounds = (0.4, 0.6)
    shape_res = 25
    c1 = cube(bounds, shape_res)
    c1[:, 1] -= 0.35
    c2 = cube(bounds, shape_res)
    x = np.concatenate((c1, c2))
    e = Engine()
    e.generate(x, model, steps, gmin, gmax, incr)


@app.command()
def train():
    save_model(train_model(make_model()))


if __name__ == "__main__":
    app()
