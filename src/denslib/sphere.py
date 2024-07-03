from __future__ import annotations
from typing import Sequence, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from tqdm import tqdm

if TYPE_CHECKING:
    from src.denslib.environment import EnvironmentMeasure


def query_ball(
    env_measure: Sequence[EnvironmentMeasure],
    pos1: npt.NDArray,
    pos2: npt.NDArray,
    radius: float,
    chunksize: int | None = None,
) -> dict[int, npt.NDArray]:
    """
    For each point in pos1, find the number of pos2 neighbors within
    the specified radius.

    """
    tree2 = KDTree(pos2)
    env_meas_dict = {}

    # Calculate all at once if no chunksize is given
    if chunksize is None:
        tree1 = KDTree(pos1)

        neighbors = tree1.query_ball_tree(tree2, r=radius)

        for j, em in enumerate(env_measure):
            env_meas_dict[j] = np.array([em.measure(inds) for inds in neighbors])

        return env_meas_dict

    # Calculate in chunks
    num_chunks = len(pos1) // chunksize

    for j in range(len(env_measure)):
        env_meas_dict[j] = np.empty(len(pos1), dtype=np.float64)

    for i in tqdm(range(num_chunks), total=num_chunks):
        low_idx = i * chunksize
        high_idx = (i + 1) * chunksize
        tree1 = KDTree(pos1[low_idx:high_idx])
        neighbors = tree1.query_ball_tree(tree2, r=radius)

        for j, em in enumerate(env_measure):
            env_meas_dict[j][low_idx:high_idx] = np.array(
                [em.measure(inds) for inds in neighbors]
            )

    if high_idx < len(pos1):
        tree1 = KDTree(pos1[high_idx:])

        neighbors = tree1.query_ball_tree(tree2, r=radius)

        for j, em in enumerate(env_measure):
            env_meas_dict[j][high_idx:] = np.array(
                [em.measure(inds) for inds in neighbors]
            )

    return env_meas_dict


def random_points(
    x: npt.NDArray, y: npt.NDArray, z: npt.NDArray, R: float, N: int
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Create N random points within a sphere of radius R centered on
    the point (x, y, z).

    """
    r = np.random.uniform(0, R, N)
    phi = np.random.uniform(0, 2 * np.pi, N)
    theta = np.random.uniform(0, np.pi, N)

    X = x + r * np.cos(phi) * np.sin(theta)
    Y = y + r * np.sin(phi) * np.sin(theta)
    Z = z + r * np.cos(theta)

    return X, Y, Z
