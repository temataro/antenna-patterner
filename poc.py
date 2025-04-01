#!/usr/bin/env python3

import numpy as np
import seaborn as sns
import scipy.signal as sp
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.constants import c, pi

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


def plot_array_factor_polar(
    array_factor, theta_values, antenna_positions, title="Antenna Array Pattern"
):
    """Made with a generous donation from Claude 3.7, thank you Claude, very cool."""

    # Add small epsilon to avoid log(0)
    magnitude_db = 20 * np.log10(np.abs(array_factor) + 0.1)
    magnitude_db = magnitude_db - np.max(magnitude_db)  # normalize

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

    ax.plot(theta_values, magnitude_db)
    # Fill the area under the curve
    ax.fill(theta_values, magnitude_db, alpha=0.25)

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("Angle θ (rad)", fontsize=12)
    ax.set_ylabel("Normalized Gain (dB)", fontsize=12, labelpad=30)

    r_elements = np.sqrt(antenna_positions[:, 0] ** 2 + antenna_positions[:, 1] ** 2)
    r_elements /= np.max(r_elements)
    theta_elements = np.arctan2(antenna_positions[:, 1], antenna_positions[:, 0])

    ax.set_ylim([-40, np.max(r_elements) + 4])

    # Plot antenna elements as points
    ax.scatter(
        theta_elements,
        r_elements,
        color="red",
        s=100,
        zorder=5,
        label="Antenna Elements",
    )

    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    return fig, ax


def main() -> None:
    R, G = load_array_elements()

    af: NDArray[np.complex64] = array_factor_1d("p0", R, G)
    print(af[100:120])
    plot_array_factor_polar(
        af,
        theta_values=np.linspace(0, 2*pi, 360),
        antenna_positions=R
    )


if __name__ == "__main__":
    main()
