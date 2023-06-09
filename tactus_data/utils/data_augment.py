import json
import random
import copy
from typing import List, Tuple, Dict
from pathlib import Path
from typing import Union, Generator

import numpy as np
import cv2
from sklearn.model_selection._search import ParameterGrid

from tactus_data.utils.skeleton import Skeleton


DEFAULT_GRID = {
    "noise_amplitude": np.linspace(1, 4, 2),
    "horizontal_flip": [True, False],
    "vertical_flip": [True, False],
    "skeleton_flip": [True, False],
    "rotation_y": np.linspace(-20, 20, 3),
    "rotation_z": np.linspace(-20, 20, 3),
    "rotation_x": np.linspace(-20, 20, 3),
    "scale_x": np.linspace(0.8, 1.2, 3),
    "scale_y": np.linspace(0.8, 1.2, 3),
}


def augment_noise_2d(keypoints: list, noise_amplitude: float, width: float, height: float) -> np.ndarray:
    """
    add noise to every keypoints of a skeleton

    Parameters
    ----------
    keypoints : list
        list of all the skeleton keypoints with only x and y coordinates
    noise_amplitude : float
        coefficient the random noise of maximum 1% of total skeleton
        amplitude is multiplied by

    Returns
    -------
    np.ndarray
        list of all the new skeleton keypoints
    """
    xscale = width / 100
    yscale = height / 100

    for i in range(0, len(keypoints), 2):
        keypoints[i] += noise_amplitude * xscale * (random.random() * 2 - 1)
        keypoints[i + 1] += noise_amplitude * yscale * (random.random() * 2 - 1)

    return keypoints


def augment_transform(keypoints: list, transform_mat: np.ndarray) -> np.ndarray:
    """
    transform a skeleton using a transformation matrix

    Parameters
    ----------
    keypoints : list
        list of all the skeleton keypoints with only x and y coordinates
    transform_mat : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        list of all the new skeleton keypoints
    """
    keypoints = np.array(keypoints, dtype="float").reshape((1, -1, 2))
    keypoints = cv2.perspectiveTransform(keypoints, transform_mat)
    return keypoints.flatten().tolist()


def transform_matrix_from_grid(
        resolution: Tuple[int, int],
        transform_dict: dict = None,
        ) -> np.ndarray:
    """
    generate the transformation matrix from a dictionnary

    Parameters
    ----------
    resolution : tuple[int, int]
        resolution of the incoming frame
    transform_dict : dict, optional
        dictionnary to create the matrix from, by default None

    Returns
    -------
    np.ndarray
        transformation matrix
    """

    return get_transform_matrix(resolution,
                                **transform_dict)


def get_transform_matrix(resolution: Tuple[int, int],
                         horizontal_flip: bool = False,
                         vertical_flip: bool = False,
                         skeleton_flip: bool = False,
                         rotation_x: float = 0,
                         rotation_y: float = 0,
                         rotation_z: float = 0,
                         scale_x: float = 1,
                         scale_y: float = 1,
                         **_
                         ):
    """Create the transform matrix using cartesian dimension"""
    # skeleton flip
    if skeleton_flip:
        horizontal_flip = not horizontal_flip
        rotation_y += 180
    # split input
    h_flip_coef = -1 if horizontal_flip else 1
    v_flip_coef = -1 if vertical_flip else 1
    t_x, t_y, t_z = (0, 0, 0)
    s_x, s_y, s_z = (h_flip_coef * scale_x, v_flip_coef * scale_y, 1)
    # degrees to rad
    theta_rx = np.deg2rad(rotation_x)
    theta_ry = np.deg2rad(rotation_y)
    theta_rz = np.deg2rad(rotation_z)
    # sin and cos
    sin_rx, cos_rx = np.sin(theta_rx), np.cos(theta_rx)
    sin_ry, cos_ry = np.sin(theta_ry), np.cos(theta_ry)
    sin_rz, cos_rz = np.sin(theta_rz), np.cos(theta_rz)

    height, width = resolution
    diag = (height ** 2 + width ** 2) ** 0.5
    # focal length
    focal = diag
    if np.sin(theta_rz) != 0:
        focal /= 2 * np.sin(theta_rz)
    # Adjust translation on z
    t_z = (focal - t_z) / s_z ** 2
    # All matrices
    # from 3D to Cartesian dimension
    M_tocart = np.array([[1, 0, -width / 2],
                         [0, 1, -height / 2],
                         [0, 0, 1],
                         [0, 0, 1]])
    # from Cartesian to 3D dimension
    M_fromcart = np.array([[focal, 0, width / 2, 0],
                           [0, focal, height / 2, 0],
                           [0, 0, 1, 0]])
    # translation matrix
    T_M = np.array([[1, 0, 0, t_x],
                    [0, 1, 0, t_y],
                    [0, 0, 1, t_z],
                    [0, 0, 0, 1]])

    # Rotation on all axes
    R_Mx = np.array([[1, 0, 0, 0],
                     [0, cos_rx, -sin_rx, 0],
                     [0, sin_rx, cos_rx, 0],
                     [0, 0, 0, 1]])
    # get the rotation matrix on y axis
    R_My = np.array([[cos_ry, 0, -sin_ry, 0],
                     [0, 1, 0, 0],
                     [sin_ry, 0, cos_ry, 0],
                     [0, 0, 0, 1]])
    # get the rotation matrix on z axis
    R_Mz = np.array([[cos_rz, -sin_rz, 0, 0],
                     [sin_rz, cos_rz, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    # final_rotation
    R_M = np.dot(np.dot(R_Mx, R_My), R_Mz)
    # Scaling matrix
    S_M = np.array([[s_x, 0, 0, 0],
                    [0, s_y, 0, 0],
                    [0, 0, s_z, 0],
                    [0, 0, 0, 1]])
    M_cart = T_M.dot(R_M).dot(S_M)
    M_final = M_fromcart.dot(M_cart).dot(M_tocart)
    return M_final


def augment_skeleton(skeleton: Skeleton,
                     matrix: np.ndarray,
                     noise_amplitude: float = 0,
                     ) -> list:
    """
    augment a single skeleton

    Parameters
    ----------
    keypoints : list
        list of all the skeleton keypoints with only x and y coordinates
    matrix : np.ndarray
        transformation matrix of size (3*3)
    noise_amplitude : float, optional
        the noise amplitude, by default 0

    Returns
    -------
    list
        the augmented skeleton
    """
    keypoints = skeleton.keypoints

    keypoints = augment_transform(keypoints, matrix)

    width = skeleton.width
    height = skeleton.height
    keypoints = augment_noise_2d(keypoints, noise_amplitude, width, height)

    skeleton = Skeleton(keypoints=keypoints)
    return skeleton


def grid_augment(formatted_json: Path,
                 grid: Union[Dict[str, list], List[Dict[str, list]]]):
    """
    augment a JSON with a grid of parameters. The result files
    are going to be written in the same folder as the original
    file.

    Parameters
    ----------
    formatted_json : Path
        the path to the JSON that is going to be augmented
    grid : dict[str, list] | list[dict[str, list]]
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.
        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See the examples below.
    """
    original_data = json.load(formatted_json.open())
    original_stem = formatted_json.stem
    suffix = formatted_json.suffix

    for i, augmented_json in enumerate(grid_augment_generator(original_data, grid)):

        new_stem = f"{original_stem}_augment_{i}"
        new_filename = formatted_json.with_name(f'{new_stem}{suffix}')
        json.dump(augmented_json, new_filename.open(mode="w"))


def grid_augment_generator(
        formatted_json: dict,
        grid: Union[Dict[str, list], List[Dict[str, list]]]
        ) -> Generator[dict, None, None]:
    """
    augment a JSON with a grid of parameters. The result dictionnaries
    are yielded.

    Parameters
    ----------
    formatted_json : dict
        a dict that contains `resolution` and `frames`
    grid : dict[str, list] | list[dict[str, list]]
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.
        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See the examples below.

    Yields
    ------
    dict
        the new augmented dict.
    """
    for params in ParameterGrid(grid):
        matrix = transform_matrix_from_grid(formatted_json["resolution"], params)

        noise_amplitude = 0
        if "noise_amplitude" in params:
            noise_amplitude = params["noise_amplitude"]

        augmented_json = copy.deepcopy(formatted_json)
        for frame in augmented_json["frames"]:
            for skeleton in frame["skeletons"]:
                if isinstance(skeleton, Dict):
                    skeleton = Skeleton(keypoints=skeleton["keypoints"])

                skeleton = augment_skeleton(skeleton,
                                            matrix,
                                            noise_amplitude)

        augmented_json["augmentation"] = params

        yield augmented_json
