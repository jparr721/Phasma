import os
from typing import Final

import taichi as ti
import typer
from loguru import logger
from tensorflow.keras.models import load_model
from tqdm import tqdm

from mpm.mpm import *
from nn.cnn import make_model, save_model, train_model

app = typer.Typer(help="p1")

_SHAPE_RES: Final[int] = 25
_RES: Final[int] = 64
_DT: Final[float] = 1e-4
_DX: Final[float] = 1 / _RES
_INV_DX: Final[float] = 1 / _DX
_MASS: Final[float] = 1.0
_VOL: Final[float] = 1.0
_E: Final[float] = 1e4
_NU: Final[float] = 0.2
_MU_0: Final[float] = _E / (2 * (1 + _NU))
_LAMBDA_0: Final[float] = _E * _NU / ((1 + _NU) * (1 - 2 * _NU))
_GRAVITY = -100.0
_MODEL = "jelly"
_STEPS = 1500

dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nn", "saved_models")
_ML_MODEL: Final[Model] = load_model(os.path.join(dirname, "cnn_model.h5"))
# _ML_MODEL = None
_USE_ML = True


def generate_cube_points(r: Tuple[float, float], res: int = 10) -> np.ndarray:
    x = np.linspace(*r, num=res)
    y = np.linspace(*r, num=res)

    all_pts = []
    for row in x:
        for col in y:
            all_pts.append([row, col])
    return np.array(all_pts, dtype=np.float64)


def advance(
    gv: np.ndarray,
    gm: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    F: np.ndarray,
    C: np.ndarray,
    Jp: np.ndarray,
    ig,
    g,
):
    global _USE_ML
    p2g(
        _INV_DX,
        _MU_0,
        _LAMBDA_0,
        _MASS,
        _DX,
        _DT,
        _VOL,
        gv,
        gm,
        x,
        v,
        F,
        C,
        Jp,
        _MODEL,
    )

    ig.append(np.concatenate((gv, gm), axis=2))
    if _USE_ML:
        nn_grid_op(_ML_MODEL, gv, gm)
    else:
        grid_op(_RES, _DX, _DT, _GRAVITY, gv, gm)
    g.append(np.concatenate((gv, gm), axis=2))

    g2p(_INV_DX, _DT, gv, x, v, F, C, Jp, _MODEL)


@app.command()
def offline_sim(
    save: bool = typer.Option(False),
    outdir: str = typer.Option("tmp"),
    use_gui: bool = typer.Option(False),
):
    bc = generate_cube_points((0.4, 0.6), _SHAPE_RES)
    bc[:, 1] -= 0.35
    tc = generate_cube_points((0.4, 0.6), _SHAPE_RES)
    x = np.concatenate((tc, bc))

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    else:
        if save:
            logger.error(f"{outdir} dir exists, remove it before saving")
            exit(1)

    xv = []
    ig = []
    g = []

    n = len(x)
    dim: Final[int] = 2

    v = np.zeros((n, dim), dtype=np.float64)
    F = np.array([np.eye(dim, dtype=np.float64) for _ in range(n)])
    C = np.zeros((n, dim, dim), dtype=np.float64)
    Jp = np.ones((n, 1), dtype=np.float64)

    # Load Initial States
    xv.append(x.copy())

    pb = tqdm(range(_STEPS))
    try:
        for _ in pb:
            pb.set_postfix({"model": _MODEL})
            gv = np.zeros((_RES, _RES, 2))
            gm = np.zeros((_RES, _RES, 1))

            advance(gv, gm, x, v, F, C, Jp, ig, g)

            xv.append(x.copy())

        if save:
            for i in range(_STEPS):
                fname = f"{outdir}/{i}/"
                os.mkdir(fname)
                np.save(f"{fname}g.npy", g[i])
                np.save(f"{fname}ig.npy", ig[i])
    except Exception as e:
        logger.error(f"Busted: {e}")

    if use_gui:
        ti.init(arch=ti.cpu)
        gui = ti.GUI()
        while gui.running and not gui.get_event(gui.ESCAPE):
            for i in range(0, len(xv), 10):
                gui.clear(0x112F41)
                gui.rect(
                    np.array((0.04, 0.04)),
                    np.array((0.96, 0.96)),
                    radius=2,
                    color=0x4FB99F,
                )
                gui.circles(xv[i], radius=1.5, color=0xED553B)
                gui.show()


@app.command()
def gen_data():
    global _GRAVITY, _MODEL, _USE_ML, _STEPS

    _USE_ML = False
    _STEPS = 4000

    gravities = [
        -20,
        -30,
        -40,
        -50,
        -60,
        -70,
        -80,
        -90,
        -100,
        -110,
        -120,
        -130,
        -140,
        -150,
        -160,
        -170,
        -180,
        -190,
        -200,
        -300,
    ]

    if not os.path.exists("datasets"):
        os.mkdir("datasets")

    for model in ("jelly",):  # "liquid", "snow"):
        for gravity in tqdm(gravities):
            _GRAVITY = gravity
            try:
                outdir = f"datasets/{model}_{gravity * -1}"
                _MODEL = model
                offline_sim(True, outdir, False)
            except Exception:
                logger.error("This sim went wrong, skipping")


@app.command()
def train():
    save_model(train_model(make_model()))


if __name__ == "__main__":
    app()
