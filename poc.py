#!/usr/bin/env python3

import numpy as np
import seaborn as sns
import scipy.signal as sp
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.constants import c, pi

from utils import plot_array_factor_polar

"""
antenna radiation pattern = array factor * element pattern

Each array element has the following properties:
    - r_n: distance from coordinate plane origin
        (represented in Cartesian coordinates)
    - g_n: complex gain factor (weights and phases)

We typically only take the pattern at theta=0 and phi=90 degrees.
Leave the rest.
"""

f: float = 10e9
k: float = 2 * pi / f
l: float = c / f
dx: float = l * 0.5
dy: float = l * 0.5


def array_factor_1d(
    theta_or_phi: str, R: NDArray[float], G: NDArray[np.complex64]
) -> NDArray[np.complex64]:
    """
    Theta or Phi has to be constant while the other is the variable we observe.
    Format: t90 means hold theta constant at 90 while phi changes from -pi to pi
            p0 means hold phi constant at 0 degrees while theta goes from -pi/2 to pi/2
    """

    if theta_or_phi[0] == "t":
        theta: float = float(theta_or_phi[1:]) * np.pi / 180
        phi: NDArray[float] = np.linspace(-pi, pi, 360) * np.pi / 180
    else:
        phi: float = float(theta_or_phi[1:]) * np.pi / 180
        theta: NDArray[float] = np.linspace(-pi / 2, pi / 2, 360) * np.pi / 180

    u: float = (np.sin(theta) * np.cos(phi)).reshape((360, 1))
    v: float = (np.sin(theta) * np.sin(phi)).reshape((360, 1))

    # weight coordinates with complex gains
    x_coords: NDArray[np.complex64] = (R[:, 0] * G).reshape((R.shape[0], 1))
    y_coords: NDArray[np.complex64] = (R[:, 1] * G).reshape((R.shape[0], 1))

    # individual phase shifts
    ps: NDArray[np.complex64] = -1j * k * (x_coords @ u.T + y_coords @ v.T)

    af: NDArray[np.complex64] = np.sum(ps, axis=0)
    af /= np.max(af)

    return af


def load_array_elements(
    filename: str | None = None,
) -> tuple[NDArray[float], NDArray[np.complex64]]:
    if filename == None:
        # dummy loading
        N: int = 11
        R: NDArray[float] = np.array([[dx * i - N // 2 * dx, 0] for i in range(N)])
        g_n: NDArray[np.complex64] = np.ones(N, dtype=np.complex64) * np.exp(1j * 0)
    else:
        raise NotImplementedError

    return R, g_n


def main() -> None:
    R, G = load_array_elements()

    af: NDArray[np.complex64] = array_factor_1d("p0", R, G)
    print(af[100:120])
    plot_array_factor_polar(
        af, theta_values=np.linspace(0, 2 * pi, 360), antenna_positions=R
    )


if __name__ == "__main__":
    main()
