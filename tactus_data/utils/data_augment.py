import json
import random
import copy
from pathlib import Path

import numpy as np
import cv2
from sklearn.model_selection._search import ParameterGrid

from tactus_data.utils.skeletonization import skeleton_bbx


DEFAULT_GRID = {
    "noise_amplitude": np.linspace(1, 4, 2),
    "horizontal_flip": [True, False],
    "rotation_y": np.linspace(-20, 20, 3),
    "rotation_z": np.linspace(-20, 20, 3),
    "rotation_x": np.linspace(-20, 20, 3),
    "scale_x": np.linspace(0.8, 1.2, 3),
    "scale_y": np.linspace(0.8, 1.2, 3),
}


def _skel_width_height(keypoints: list):
    """Used by noise_2d(): Returns the max width and height of
       skeletons keypoints
    """
    _, _, width, height = skeleton_bbx(keypoints)
    xscale = width / 100
    yscale = height / 100
    return xscale, yscale


def augment_noise_2d(keypoints: list, noise_amplitude: float) -> np.ndarray:
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
    xscale, yscale = _skel_width_height(keypoints)

    for i in range(0, len(keypoints), 2):
        keypoints[i] += noise_amplitude * xscale * (random.random()*2-1)
        keypoints[i+1] += noise_amplitude * yscale * (random.random()*2-1)

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
    return keypoints.flatten()


def transform_matrix_from_grid(
        resolution: tuple[int, int],
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


def get_transform_matrix(resolution: tuple[int, int],
                         horizontal_flip: bool = False,
                         vertical_flip: bool = False,
                         rotation_x: float = 0,
                         rotation_y: float = 0,
                         rotation_z: float = 0,
                         scale_x: float = 1,
                         scale_y: float = 1,
                         **_
                         ):
    """Create the transform matrix using cartesian dimension"""
    # split input
    h_flip_coef = -1 if horizontal_flip else 1
    v_flip_coef = -1 if vertical_flip else 1
    t_x, t_y, t_z = (0, 0, 0)
    s_x, s_y, s_z = (h_flip_coef*scale_x, v_flip_coef*scale_y, 1)
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


def augment_skeleton(keypoints: list,
                     matrix: np.ndarray,
                     noise_amplitude: float = 0,
                     ) -> list:
    keypoints = augment_transform(keypoints, matrix)
    keypoints = augment_noise_2d(keypoints, noise_amplitude)

    return keypoints.tolist()


def grid_augment(formatted_json: Path,
                 grid: dict):
    original_data = json.load(formatted_json.open())
    original_stem = formatted_json.stem

    for i, params in enumerate(ParameterGrid(grid)):
        matrix = transform_matrix_from_grid(original_data["resolution"], params)
        if "noise_amplitude" in params:
            noise_amplitude = params["noise_amplitude"]

        augmented_json = copy.deepcopy(original_data)
        for frame in augmented_json["frames"]:
            for skeleton in frame["skeletons"]:
                skeleton["keypoints"] = augment_skeleton(skeleton["keypoints"],
                                                         matrix,
                                                         noise_amplitude)

        augmented_json["augmentation"] = params

        new_filename = formatted_json.with_stem(f"{original_stem}_augment_{i}")
        json.dump(augmented_json, new_filename.open(mode="w"))
