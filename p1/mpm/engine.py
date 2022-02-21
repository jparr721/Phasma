import os
from functools import partial
from typing import Final, List, Tuple

import numpy as np
from loguru import logger
from tensorflow.keras.models import Model, load_model
from tqdm import tqdm

from mpm.mpm import apply_boundary_conditions, g2p, grid_op, p2g


def cube(bounds: Tuple[float, float], res=10) -> np.ndarray:
    x = np.linspace(*bounds, num=res)
    return np.array([[row, col] for row in x for col in x])


class Engine(object):
    def __init__(self, outdir="tmp", use_ml=False):
        self.outdir = outdir
        self.use_ml = use_ml

        # How-swappable functor types
        self.p2g = p2g
        self.g2p = g2p
        self.grid_op = grid_op
        self.apply_boundary_conditions = partial(apply_boundary_conditions, "sticky")

        if self.use_ml:
            ml_model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "nn",
                "saved_models",
                "cnn_model.h5",
            )
            self.ml_model: Final[Model] = load_model(ml_model_path)

        # Final sim constants
        self.grid_res: Final[int] = 64
        self.dt: Final[float] = 1e-4
        self.dx: Final[float] = 1 / self.grid_res
        self.inv_dx: Final[float] = 1 / self.dx
        self.mass: Final[float] = 1.0
        self.vol: Final[float] = 1.0

    def simulate(
        self,
        x: np.ndarray,
        *,
        saved=[],
        use_gui=True,
        model="jelly",
        steps=2000,
        gravity=-200.0,
        E=1e3,
        nu=0.2,
    ):
        self.x = x
        _n = len(self.x)
        dim = 2
        self.v = np.zeros((_n, dim), dtype=np.float64)
        self.F = np.array([np.eye(dim, dtype=np.float64) for _ in range(_n)])
        self.C = np.zeros((_n, dim, dim), dtype=np.float64)
        self.Jp = np.ones((_n, 1), dtype=np.float64)
        mu_0: Final[float] = E / (2 * (1 + nu))
        lambda_0: Final[float] = E * nu / ((1 + nu) * (1 - 2 * nu))

        xv = [self.x.copy()]
        try:
            for _ in tqdm(range(steps), postfix={"model": model}):
                self.gm = np.zeros((self.grid_res, self.grid_res, 1))
                self.gv = np.zeros((self.grid_res, self.grid_res, 2))

                self.p2g(
                    self.inv_dx,
                    mu_0,
                    lambda_0,
                    self.mass,
                    self.dx,
                    self.dt,
                    self.vol,
                    self.gv,
                    self.gm,
                    self.x,
                    self.v,
                    self.F,
                    self.C,
                    self.Jp,
                    model,
                )

                self.grid_op(self.dx, self.dt, gravity, self.gv, self.gm)
                self.apply_boundary_conditions(self.gv, self.gm)

                self.g2p(
                    self.inv_dx,
                    self.dt,
                    self.gv,
                    self.x,
                    self.v,
                    self.F,
                    self.C,
                    self.Jp,
                    model,
                )

                xv.append(self.x.copy())

            if len(saved) > 0:
                self._unload(*saved)
        except Exception as e:
            logger.error(f"Sim crashed: {e}")

        if use_gui:
            import taichi as ti

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

    def generate(
        self,
        x: np.ndarray,
        model: str,
        steps: int,
        gmin: int,
        gmax: int,
        incr: int,
    ):
        # Hardcoded for now to avoid annihilating our hdd
        saved = ["igs", "gs", "igbc", "gbc"]
        for gravity in tqdm(range(gmin, gmax, incr)):
            self.outdir = os.path.join("datasets", f"{model}_{gravity * -1}")
            self.simulate(
                x, saved=saved, use_gui=False, model=model, steps=steps, gravity=gravity
            )

    def _unload(self, *names):
        assert len(names) > 0
        entries = len(self.__dict__[names[0]])
        for i in range(entries):
            name = os.path.join(self.outdir, str(i))
            os.mkdir(name)
            for name in names:
                assert isinstance(self.__dict__[name], np.ndarray)
                np.save(f"{name}.npy", self.__dict__[name])
