import os
from typing import List

import typer

from mpm.engine import Engine, cube
from mpm.mpm import *
from nn.cnn import make_model, save_model, train_model

app = typer.Typer(help="p1")


@app.command()
def offline_sim(
    outdir: str = typer.Option("tmp"),
    use_gui: bool = typer.Option(False),
):
    bounds = (0.4, 0.6)
    shape_res = 25
    c1 = cube(bounds, shape_res)
    c1[:, 0] -= 0.15
    c1[:, 1] -= 0.30
    c2 = cube(bounds, shape_res)

    x = np.concatenate((c1, c2))
    e = Engine(outdir)
    e.simulate(
        x,
        use_gui=use_gui,
        model="jelly",
        boundary_ops="slip",
    )


@app.command()
def gen_data(
    model: str = typer.Option("jelly"),
    steps: int = typer.Option(4000),
    gmin: int = typer.Option(-10),
    gmax: int = typer.Option(-300),
    incr: int = typer.Option(10),
):
    datasets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    e = Engine(outdir=datasets)

    bounds = (0.4, 0.6)
    shape_res = 25
    c1 = cube(bounds, shape_res)
    c1[:, 0] -= 0.15
    c1[:, 1] -= 0.30
    c2 = cube(bounds, shape_res)

    x = np.concatenate((c1, c2))
    e.generate(x, model, steps, gmin, gmax, incr)


@app.command()
def train():
    save_model(train_model(make_model()))


if __name__ == "__main__":
    app()
