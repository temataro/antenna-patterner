#!/usr/bin/env python3

"""
Collection of helper tools that make more sense
to write in Python.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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
