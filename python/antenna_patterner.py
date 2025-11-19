""" """

import numpy as np
import matplotlib.pyplot as plt

from utils import *

# 1 unit cell = lambda/20
c = 299_792_458  # m/s
f = 1e9  # Hz
epsilon_r = 1
lam = c / (np.sqrt(epsilon_r) * f)  # m

#                        x    y    z
cells_per_lambda = 20  # lambdas
num_cells = cells_per_lambda * 10
grid = np.zeros(shape=(num_cells, num_cells, num_cells), dtype=np.float32)

# Initially, we'll just work on a slice of this grid: the xy plane
xy = grid[:, :, num_cells // 2]

# Let's make a linear array at lambda / 2 spacing
xy[::20, num_cells // 2] = 1
# xy[num_cells // 2, ::10] = 1
xy[::5, ::20] = 1
x_coords, y_coords = np.nonzero(xy)
print(f"There are {len(x_coords)} radiators.")

plot_geometry_2d(xy)


def get_r(x, y, z, target):
    return np.array(
        [
            lam * (x - target[0]),
            lam * (y - target[1]),
            lam * (z - target[2]),
        ]
    )


def get_r_arr(x, y, z, target):
    # For each x, y, z find the vector r
    N = len(x)
    R = np.zeros((N, 3), dtype=np.float32)

    i = 0
    for x_coord, y_coord, z_coord in zip(x, y, z):
        R[i] = np.array(
            [
                lam * (x_coord - target[0]),
                lam * (y_coord - target[1]),
                lam * (z_coord - target[2]),
            ]
        )
        i += 1

    return R


# plot_geometry_2d(xy)
# Now we compute the array pattern in one slice at far field.
# say our target is 20 lambda away on the same plane as the array
r = 200 * cells_per_lambda
thetas = 720

target_coords = np.zeros((thetas, 3), dtype=np.float32)
theta_values = np.linspace(start=0, stop=2 * np.pi, num=thetas)
for i, phi in enumerate(theta_values):
    target_coords[i] = np.array([r * np.cos(phi), r * np.sin(phi), num_cells // 2])

AF = np.zeros((thetas), dtype=np.complex64)


def k_vec(theta, phi):
    # accepts angles in radians
    return np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    ) * (2 * np.pi / lam)


k = 2 * np.pi / lam
k_v = k_vec(theta=0, phi=np.pi / 2)
# k_v = np.array([1, 0, 0]) * k
for t, target in enumerate(target_coords):
    for x, y in zip(x_coords, y_coords):
        # AF[t] += np.exp(1j * np.cos(t * 2 * np.pi / thetas) * k * get_r(x, 0, 0, target)[0])
        AF[t] += np.exp(
            1j
            * np.dot(k_vec(phi=0, theta=t * 2 * np.pi / thetas), get_r(x, y, 0, target))
        )

plot_AF(AF, theta_values)
