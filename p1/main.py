import os
from typing import Final

import numba as nb
import taichi as ti
import typer
from loguru import logger
from tqdm import tqdm

from mpm.mpm import *

app = typer.Typer(help="p1")

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
_GRAVITY = -100.0
_MODEL = "jelly"
_STEPS: Final[int] = 3000


def generate_cube_points(r: Tuple[float, float], res: int = 10) -> np.ndarray:
    x = np.linspace(*r, num=res)
    y = np.linspace(*r, num=res)

    all_pts = []
    for row in x:
        for col in y:
            all_pts.append([row, col])
    return np.array(all_pts, dtype=np.float64)


@nb.njit
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


def sim(save: bool, outdir: str, use_gui=False):
    gui = ti.GUI()
    # cp = generate_cube_points((0.4, 0.6), _SHAPE_RES)
    # cp[:, 1] -= 0.35
    # x = np.concatenate((generate_cube_points((0.4, 0.6), _SHAPE_RES), cp))
    x = generate_cube_points((0.4, 0.6), _SHAPE_RES)

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    else:
        if save:
            logger.error(f"{outdir} dir exists, remove it before saving")
            exit(1)

    xv = []
    vv = []
    Fv = []
    Cv = []
    Jpv = []
    gvv = []
    gmv = []

    n = len(x)
    dim: Final[int] = 2

    v = np.zeros((n, dim), dtype=np.float64)
    F = np.array([np.eye(dim, dtype=np.float64) for _ in range(n)])
    C = np.zeros((n, dim, dim), dtype=np.float64)
    Jp = np.ones((n, 1), dtype=np.float64)

    # Load Initial States
    xv.append(x.copy())
    vv.append(v.copy())
    Fv.append(F.copy())
    Cv.append(C.copy())
    Jpv.append(Jp.copy())

    pb = tqdm(range(_STEPS))
    for _ in pb:
        pb.set_postfix({"model": _MODEL})
        gv = np.zeros(((_RES + 1), (_RES + 1), 2))
        gm = np.zeros(((_RES + 1), (_RES + 1), 1))
        advance(gv, gm, x, v, F, C, Jp)

        xv.append(x.copy())
        vv.append(v.copy())
        Fv.append(F.copy())
        Cv.append(C.copy())
        Jpv.append(Jp.copy())
        gvv.append(gv.copy())
        gmv.append(gm.copy())

    if save:
        for i in range(_STEPS):
            fname = f"{outdir}/{i}/"
            os.mkdir(fname)
            np.save(f"{fname}x.npy", xv[i])
            np.save(f"{fname}v.npy", vv[i])
            np.save(f"{fname}F.npy", Fv[i])
            np.save(f"{fname}C.npy", Cv[i])
            np.save(f"{fname}Jp.npy", Jpv[i])
            np.save(f"{fname}gv.npy", gvv[i])
            np.save(f"{fname}gm.npy", gmv[i])

    if use_gui:
        ti.init(arch=ti.gpu)
        while gui.running and not gui.get_event(gui.ESCAPE):
            for xx in xv:
                gui.clear(0x112F41)
                gui.rect(
                    np.array((0.04, 0.04)),
                    np.array((0.96, 0.96)),
                    radius=2,
                    color=0x4FB99F,
                )
                gui.circles(xx, radius=1.5, color=0xED553B)
                gui.show()


if __name__ == "__main__":
    gravities = [-9.8, -25, -50, -100, -200, -500]
    models = [("jelly", "snow", "liquid") * len(gravities)]

    it = 0
    for gravity, models in tqdm(zip(gravities, models)):
        _GRAVITY = gravity
        for model in models:
            try:
                outdir = f"{model}_{it}"
                _MODEL = model
                sim(True, outdir)
            except Exception as e:
                logger.error("This sim went wrong, skipping")

            it += 1
