#!/usr/bin/env python3

"""
Collection of helper tools that make more sense
to write in Python.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_AF(array_factor, theta_values):
    magnitude_db = 20 * np.log10(np.abs(array_factor)) # + 1e-9)
    magnitude_db = magnitude_db - np.max(magnitude_db)  # normalize

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

    ax.plot(theta_values, magnitude_db, c="#5400FF")

    ax.set_xlabel("Angle θ (rad)", fontsize=12)
    ax.set_ylabel("Normalized Gain (dB)", fontsize=12, labelpad=30)
    ax.set_ylim([-45, 5])
    plt.show()

def plot_geometry_2d(grid):
    x = len(grid[:, 0])
    y = len(grid[0, :])
    x_coords, y_coords = np.nonzero(grid)

    plt.scatter(x_coords, y_coords, c="#5400FF", alpha=0.3, s=50)
    plt.xlim([-10, x + 10])
    plt.ylim([-10, y + 10])
    plt.grid(True)
    plt.show()

