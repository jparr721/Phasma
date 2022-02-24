import os
from typing import Final, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
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
        self.apply_boundary_conditions = apply_boundary_conditions

        if self.use_ml:
            logger.info("Loading ML Model")
            ml_model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
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
        use_gui=True,
        model="jelly",
        boundary_ops="sticky",
        steps=2000,
        gravity=-300.0,
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

        xv = []
        self.igs = []
        self.gs = []
        self.igbc = []
        self.gbc = []

        self.current_step = 0
        try:
            for self.current_step in tqdm(range(steps), postfix={"model": model}):
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

                if not self.use_ml:
                    self.igs.append(self.gv.copy())
                    self.grid_op(self.dx, self.dt, gravity, self.gv, self.gm)
                    self.gs.append(self.gv.copy())

                    self.igbc.append(self.gv.copy())
                    self.apply_boundary_conditions(self.gv, boundary_ops)
                    self.gbc.append(self.gv.copy())
                else:
                    self.ml_grid_boundary(self.gv)

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

                if use_gui:
                    xv.append(self.x.copy())
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
                    # if not os.path.exists("pics"):
                    #     os.mkdir("pics")
                    #     os.mkdir("pics/x/")
                    #     os.mkdir("pics/y/")
                    # plt.imsave(f"pics/x/x{i}.png", self.gbc[i][:, :, 0])
                    # plt.imsave(f"pics/y/y{i}.png", self.gbc[i][:, :, 1])

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
        saved: Final[List[str]] = ["igs", "gbc"]

        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
        else:
            raise RuntimeError("Remove the datasets directory to continue")

        for gravity in tqdm(range(gmax, gmin, incr)):
            data_dir = os.path.join(self.outdir, f"{model}_{gravity}")

            if not os.path.exists(data_dir):
                os.mkdir(data_dir)
            else:
                raise RuntimeError(f"Dirname {data_dir} already exists")

            self.simulate(
                x.copy(), use_gui=False, model=model, steps=steps, gravity=gravity
            )
            self._unload(data_dir, *saved)

    def ml_grid_boundary(self, gv: np.ndarray):
        assert self.use_ml
        mask = (gv[:, :, 0] != 0) | (gv[:, :, 1] != 0)
        model_input = np.concatenate((gv, np.expand_dims(mask, axis=2)), axis=2)
        grid = np.squeeze(self.ml_model.predict(np.expand_dims(model_input, axis=0)))
        gv[:, :, :2] = np.concatenate(
            (
                np.expand_dims(grid[:, :, 0], axis=2),
                np.expand_dims(grid[:, :, 1], axis=2),
            ),
            axis=2,
        )
        # mask = grid[:, :, 2]
        # if os.path.exists("pics"):
        #     plt.imsave(f"pics/mask/mask_{self.current_step}.png", mask)
        #     plt.imsave(f"pics/x/x_{self.current_step}.png", grid[:, :, 0])
        #     plt.imsave(f"pics/y/y_{self.current_step}.png", grid[:, :, 1])

    def _unload(self, root_dir: str, *names):
        entries = len(self.__dict__[names[0]])
        logger.info(f"Saving {entries} entries")
        for i in range(entries):
            subdir = os.path.join(root_dir, str(i))

            if not os.path.exists(subdir):
                os.mkdir(subdir)
            else:
                raise RuntimeError(f"Dirname {subdir} already exists")

            for name in names:
                d = os.path.join(subdir, name)
                np.save(f"{d}.npy", self.__dict__[name][i])
