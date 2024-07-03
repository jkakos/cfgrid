from __future__ import annotations
from typing import Sequence, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from tqdm import tqdm

if TYPE_CHECKING:
    from src.denslib.environment import EnvironmentMeasure


def find_all_neighbors(
    env_measure: Sequence[EnvironmentMeasure],
    rp_max: float,
    rpi_max: float,
    pos1: npt.NDArray,
    pos2: npt.NDArray,
    ang_pos1: npt.NDArray | None = None,
    ang_pos2: npt.NDArray | None = None,
    chunksize: int = 1000,
) -> dict[int, npt.NDArray]:
    """
    Loop through all galaxies to find neighbors bounded by rp_max and rpi_max.

    """
    if ang_pos1 is not None:
        tree2 = KDTree(ang_pos2)
    else:
        tree2 = KDTree(pos2)

    env_meas_dict = {}

    size = len(pos1)
    num_chunks = size // chunksize

    for j in range(len(env_measure)):
        env_meas_dict[j] = np.empty(len(pos1), dtype=np.float64)

    # First quickly find neighbors within a sphere, then search for
    # cylindrical neighbors within the spherical selection.
    for i in tqdm(range(num_chunks), total=num_chunks):
        low_idx = i * chunksize
        high_idx = (i + 1) * chunksize

        env_in_vol = _find_sphere_neighbors(
            env_measure,
            pos1,
            pos2,
            tree2,
            rp_max,
            rpi_max,
            low_idx=low_idx,
            high_idx=high_idx,
        )

        for j, e_in_v in env_in_vol.items():
            env_meas_dict[j][low_idx:high_idx] = e_in_v

    if high_idx < size:
        env_in_vol = _find_sphere_neighbors(
            env_measure,
            pos1,
            pos2,
            tree2,
            rp_max,
            rpi_max,
            low_idx=high_idx,
            high_idx=size,
        )

        for j, e_in_v in env_in_vol.items():
            env_meas_dict[j][high_idx:] = e_in_v

    return env_meas_dict


def _find_sphere_neighbors(
    env_measure: Sequence[EnvironmentMeasure],
    pos1: npt.NDArray,
    pos2: npt.NDArray,
    tree2: KDTree,
    rp_max: float,
    rpi_max: float,
    low_idx: int,
    high_idx: int,
) -> dict[int, npt.NDArray]:
    """
    Helper function for 'find_all_neighbors'.

    """
    tree1 = KDTree(pos1[low_idx:high_idx])
    env_meas_dict = {}
    chunksize = high_idx - low_idx

    # Ensure the sphere is large enough to encompass whole cylinder
    radius = np.sqrt(rp_max**2 + rpi_max**2)

    for j, em in enumerate(env_measure):
        env_meas_dict[j] = np.empty(chunksize, dtype=np.float64)

    # Find neighboring galaxies within sphere
    neighbors = tree1.query_ball_tree(tree2, r=radius)

    # Find neighboring galaxies within sphere AND cylinder defined
    # by rp_max and rpi_max
    for i, (idx, neighbor) in enumerate(zip(range(low_idx, high_idx), neighbors)):
        inds = np.array(neighbor)

        if len(inds) == 0:
            for j, em in enumerate(env_measure):
                env_meas_dict[j][i] = 0.0

            continue

        neighbors_in_vol_cond = find_cyl_neighbors(
            pos1[idx], pos2[inds], rp_max, rpi_max
        )

        selected_inds = inds[neighbors_in_vol_cond]
        for j, em in enumerate(env_measure):
            env_meas_dict[j][i] = em.measure(selected_inds)

    return env_meas_dict


def find_cyl_neighbors(
    pos1: npt.NDArray,
    pos2: npt.NDArray,
    rp_max: float,
    rpi_max: float,
) -> npt.NDArray:
    """
    Find the index and/or angular separation of the kth nearest
    neighbor of a galaxy projected over some distance. The X1
    arguments should correspond to a single galaxy and the X2
    arguments should be arrays of all the galaxies.

    Parameters
    ----------
    pos1 : array shape (N,)
        Cartesian coordinates of the chosen galaxy.

    pos2 : array shape (M,N)
        Cartesian coordinates of the potential neighboring galaxies.

    rp_max, rpi_max : float, float
        Maximum distances in the perpendicular and parallel
        directions to consider neighbors.

    Returns
    -------
    neighbors_in_vol : array[bool] shape (M,)
        Boolean array designating which galaxies are within the bounds
        of the chosen galaxy's position.

    """
    r_p, r_pi = separation_components(pos1, pos2)

    perpendicular_trim = np.abs(r_p) < rp_max
    parallel_trim = np.abs(r_pi) < rpi_max

    neighbors_in_vol = perpendicular_trim & parallel_trim

    return neighbors_in_vol


def separation_components(
    pos1: npt.NDArray, pos2: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Find the distance between pos1 and pos2 by breaking into
    components that are perpendicular to and parallel to the line
    of sight of pos1.

    Parameters
    ----------
    pos1, pos2 : array shape (N,), array shape (M,N)
        Cartesian coordinates to the galaxies.

    Returns
    -------
    r_p, r_pi : array shape (M,), array shape (M,)
        The line of sight and projected distances between
        pos1 and pos2.

    """
    s = pos1 - pos2
    l = 0.5 * (pos1 + pos2)

    dot = lambda x, y: np.sum(x * y, axis=1)

    r_pi = dot(s, l) / np.linalg.norm(l, axis=1)

    # In some cases with small angles, negative values may appear
    # in the sqrt function of r_p.  In such instances, this will
    # catch those errors and set r_p to 0 which is equivalent
    # for the purposes of density estimates.
    np.seterr(invalid='raise')  # force numpy to properly raise errors
    try:
        r_p = np.sqrt(dot(s, s) - (r_pi * r_pi))
    except FloatingPointError:
        np.seterr(invalid='ignore')
        r_p = np.sqrt(dot(s, s) - (r_pi * r_pi))
        r_p[np.isnan(r_p)] = 0

    return r_p, r_pi


def random_points(
    x: float, y: float, z: float, R: float, L: float, N: int
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Create N random points within a cylinder of radius R and
    total length 2L centered on the point (x, y, z).

    """
    r = np.random.uniform(0, R, N)
    phi = np.random.uniform(0, 2 * np.pi, N)
    z_cyl = np.random.uniform(-L, L, N)

    X = r * np.cos(phi)
    Y = r * np.sin(phi)
    Z = z_cyl

    theta = angle_btw_vectors(np.array([x, y, z]), np.array([0, 0, 1]))
    rot_axis = np.cross(np.array([x, y, z]), np.array([0, 0, 1]))
    points = rotate_vector_about_axis(np.array([X, Y, Z]).T, rot_axis, -theta)

    X_rot = points[:, 0] + x
    Y_rot = points[:, 1] + y
    Z_rot = points[:, 2] + z

    return X_rot, Y_rot, Z_rot


def unit_vector(v: npt.NDArray) -> npt.NDArray:
    """
    Find the unit vector of vector v.

    """
    return v / np.linalg.norm(v)


def angle_btw_vectors(v1: npt.NDArray, v2: npt.NDArray) -> float:
    """
    Find the angle between vectors v1 and v2.

    """
    uv1 = unit_vector(v1)
    uv2 = unit_vector(v2)
    theta = np.arccos(np.dot(uv1, uv2))

    return theta


def rotate_vector_about_axis(
    v1: npt.NDArray, v2: npt.NDArray, theta: float
) -> npt.NDArray:
    """
    Rotate vector v1 an angle theta about the axis v2
    (Rodrigues rotation formula).

    """
    uv2 = unit_vector(v2)

    return (
        v1 * np.cos(theta)
        + np.cross(uv2, v1) * np.sin(theta)
        + np.array([c * uv2 for c in np.sum(uv2 * v1, axis=1)]) * (1 - np.cos(theta))
    )
