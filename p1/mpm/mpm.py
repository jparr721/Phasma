from typing import Final

import numba as nb
import numpy as np

from .utils import *

_LIQUID_HARDENING: Final[float] = 1.0
_JELLY_HARDENING: Final[float] = 0.7
_SNOW_HARDENING: Final[float] = 10.0


@nb.njit
def p2g(
    inv_dx: float,
    mu_0: float,
    lambda_0: float,
    mass: float,
    dx: float,
    dt: float,
    volume: float,
    gv: np.ndarray,
    gm: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    F: np.ndarray,
    C: np.ndarray,
    Jp: np.ndarray,
    model: str = "jelly",
):
    for p in nb.prange(len(x)):
        bc = (x[p] * inv_dx - 0.5).astype(np.int64)
        if oob(bc, gv.shape[0]):
            print("p2g bc", bc)
            raise RuntimeError
        fx = (x[p] * inv_dx - bc).astype(np.float64)

        w = quadric_kernel(fx)

        mu, lambda_ = (
            constant_hardening(mu_0, lambda_0, _JELLY_HARDENING)
            if model == "jelly"
            else snow_hardening(mu_0, lambda_0, _SNOW_HARDENING, Jp[p])
            if model == "snow"
            else constant_hardening(mu_0, lambda_0, _LIQUID_HARDENING)
        )

        affine = first_piola_stress(F[p], inv_dx, mu, lambda_, dt, volume, mass, C[p])

        for i in nb.prange(3):
            for j in nb.prange(3):
                if oob(bc, gv.shape[0], np.array((i, j))):
                    print("p2g bc", bc)
                    raise RuntimeError

                dpos = (np.array((i, j)) - fx) * dx
                weight = w[i][0] * w[j][1]

                gv[bc[0] + i, bc[1] + j] += weight * (v[p] * mass + affine @ dpos)
                gm[bc[0] + i, bc[1] + j] += weight * mass


@nb.njit
def g2p(
    inv_dx: float,
    dt: float,
    gv: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    F: np.ndarray,
    C: np.ndarray,
    Jp: np.ndarray,
    model: str = "jelly",
):
    for p in nb.prange(len(x)):
        bc = (x[p] * inv_dx - 0.5).astype(np.int64)
        if oob(bc, gv.shape[0]):
            print("g2p bc", bc)
            raise RuntimeError
        fx = x[p] * inv_dx - (bc).astype(np.float64)

        w = quadric_kernel(fx)

        C[p] = 0.0
        v[p] = 0.0

        for i in range(3):
            for j in range(3):
                if oob(bc, gv.shape[0], np.array((i, j))):
                    print("g2p bc", bc)
                    raise RuntimeError
                dpos = np.array((i, j)) - fx
                grid_v = gv[bc[0] + i, bc[1] + j]
                weight = w[i][0] * w[j][1]
                v[p] += weight * grid_v
                C[p] += 4 * inv_dx * np.outer(weight * grid_v, dpos)

        # Advection
        x[p] += dt * v[p]
        F_ = (np.eye(2) + dt * C[p]) @ F[p]

        if model != "jelly":

            if model == "snow":
                U, sig, V = np.linalg.svd(F_)
                sig = np.clip(sig, 1.0 - 2.5e-2, 1.0 + 7.5e-3)
                sig = np.eye(2) * sig

                old_J = np.linalg.det(F_)
                F_ = U @ sig @ V.T
                Jp[p] = np.clip(Jp[p] * old_J / np.linalg.det(F_), 0.6, 20.0)

            if model == "liquid":
                U, sig, V = np.linalg.svd(F_)
                J = 1.0
                for dd in range(2):
                    J *= sig[dd]
                F_ = np.eye(2)
                F_[0, 0] = J

        F[p] = F_


@nb.njit
def grid_op(dx: float, dt: float, gravity: float, gv: np.ndarray, gm: np.ndarray):
    """Grid normalization and gravity application, this also handles the collision
    scenario which, right now, is "STICKY", meaning the velocity is set to zero during
    collision scenarios.

    Args:
        dx (float): dx
        dt (float): dt
        gravity (float): gravity
        gv (np.ndarray): grid velocity
        gm (np.ndarray): grid mass
    """
    v_allowed: Final[float] = dx * 0.9 / dt
    for i in range(gv.shape[0]):
        for j in range(gv.shape[1]):
            if gm[i, j][0] > 0:
                gv[i, j] /= gm[i, j][0]
                gv[i, j][1] += dt * gravity
                gv[i, j] = np.clip(gv[i, j], -v_allowed, v_allowed)


@nb.njit
def apply_boundary_conditions(gv: np.ndarray, axis_op="sticky", lower_boundary=3):
    upper_boundary = gv.shape[0] - lower_boundary

    def colliding(i):
        return i < lower_boundary or i >= upper_boundary

    def separate_slip():
        friction = 0.5
        normals = [
            unit_vector(0),
            unit_vector(0) * -1,
            unit_vector(1),
            unit_vector(1) * -1,
        ]

        for i in range(gv.shape[0]):
            for j in range(gv.shape[1]):
                for normal in normals:
                    if colliding(i) or colliding(j):
                        v = gv[i, j]
                        normal_component = np.dot(normal, v)

                        # Ref mpm.graphics section 12
                        if axis_op == "slip":
                            v -= normal * normal_component
                        elif axis_op == "separate":
                            v -= normal * min(normal_component, 0.0)
                        else:
                            raise ValueError

                        # Friction application
                        if normal_component < 0 and np.linalg.norm(v) > 1e-30:
                            v = normalized(v) * max(
                                0, np.linalg.norm(v) + normal_component * friction
                            )

                        gv[i, j] = v

    def sticky():
        gv[0:lower_boundary, :] = 0.0
        gv[:, 0:lower_boundary] = 0.0
        gv[upper_boundary:, :] = 0.0
        gv[:, upper_boundary:] = 0.0

    if axis_op == "sticky":
        sticky()
    elif axis_op == "separate" or axis_op == "slip":
        separate_slip()
    else:
        raise ValueError
