import os
from typing import Final

import taichi as ti
import typer
from tqdm import tqdm

from mpm.mpm import *

app = typer.Typer(help="p1")

ti.init(arch=ti.gpu)

_SHAPE_RES: Final[int] = 50
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
_GRAVITY: Final[float] = -100.0
_MODEL = "snow"


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
):
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
    grid_op(_RES, _DX, _DT, _GRAVITY, gv, gm)
    g2p(_INV_DX, _DT, gv, x, v, F, C, Jp, _MODEL)


@app.command()
def sim(save: bool = typer.Option(False)):
    gui = ti.GUI()
    cp = generate_cube_points((0.4, 0.6), _SHAPE_RES)
    cp[:, 1] -= 0.35
    x = np.concatenate((generate_cube_points((0.4, 0.6), _SHAPE_RES), cp))
    n = len(x)
    dim: Final[int] = 2

    v = np.zeros((n, dim), dtype=np.float64)
    F = np.array([np.eye(dim, dtype=np.float64) for _ in range(n)])
    C = np.zeros((n, dim, dim), dtype=np.float64)
    Jp = np.ones((n, 1), dtype=np.float64)

    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    p = [x.copy()]
    for i in tqdm(range(2000)):
        fname = f"tmp/{i}/"
        os.mkdir(fname)

        if save:
            np.save(f"{fname}x.npy", x)
            np.save(f"{fname}v.npy", v)
            np.save(f"{fname}F.npy", F)
            np.save(f"{fname}C.npy", C)
            np.save(f"{fname}Jp.npy", Jp)

        gv = np.zeros(((_RES + 1), (_RES + 1), 2))
        gm = np.zeros(((_RES + 1), (_RES + 1), 1))
        advance(gv, gm, x, v, F, C, Jp)

        if save:
            np.save(f"{fname}gv.npy", gv)
            np.save(f"{fname}gm.npy", gm)

        p.append(x.copy())

    while gui.running and not gui.get_event(gui.ESCAPE):
        for xx in p:
            gui.clear(0x112F41)
            gui.rect(
                np.array((0.04, 0.04)), np.array((0.96, 0.96)), radius=2, color=0x4FB99F
            )
            gui.circles(xx, radius=1.5, color=0xED553B)
            gui.show()


if __name__ == "__main__":
    app()
