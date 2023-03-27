import json
import random
from itertools import product
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2


class BK(Enum):
    Nose = 0
    LEye = 1
    REye = 2
    LEar = 3
    REar = 4
    LShoulder = 5
    RShoulder = 6
    LElbow = 7
    RElbow = 8
    LWrist = 9
    RWrist = 10
    LHip = 11
    RHip = 12
    LKnee = 13
    RKnee = 14
    LAnkle = 15
    RAnkle = 16

    list_link = [(RAnkle, RKnee),
                 (LAnkle, LKnee),
                 (RKnee, RHip),
                 (LKnee, LHip),
                 (RHip, LHip),
                 (RHip, RShoulder),
                 (LHip, LShoulder),
                 (RShoulder, LShoulder),
                 (RShoulder, RElbow),
                 (RElbow, RWrist),
                 (LShoulder, LElbow),
                 (LElbow, LWrist),
                 (Nose, LEye),
                 (Nose, REye),
                 (LEye, LEar),
                 (REye, REar)]


class gridParam:
    """
    In grid param, each element are a list of parameters, one parameter
    can be a list if there are multiple values to test. There is an
    exception for scaling where you just put all the possibilities you want to do
    """

    def __init__(self, noise=([1], [4]), translation=([0], [0], [0]),
                 rotation=([0, 20, -20], [0, 180, 30, -30], [0, 10, -10]),
                 scaling=([1, 1, 1], [1.1, 1, 1], [0.9, 1, 1], [1, 1.1, 1], [1, 0.9, 1])):
        self.noise = noise
        self.translation = translation
        self.rotation = rotation
        self.scaling = scaling


def plot_skeleton_2d(path_json: Path,
                     path_frame: Path,
                     frame_number: int = 0,
                     show_frame: bool = True):
    """
    Plot the 2D skeleton (keypoint and limbs) on top of the
    corresponding frame for testing purpose on the different
    augment

    Parameters
    ----------
    path_json : Path,
                path where the json file is located
    path_frame : Path,
                 path where the frame file is located
    frame_number : int,
                   Number of the frame you want to plot
    show_frame : bool,
                 If false allows you to see only skeleton
    """
    with open(path_json) as file:
        data = json.load(file)
    fig, ax = plt.subplots()
    max_frame = len(data['frames']) - 1

    if frame_number > max_frame:
        frame_number = max_frame
    for skeletons in data['frames'][frame_number]['skeletons']:
        keypoints = skeletons["keypoints"]
        keypoints_x = []
        keypoints_y = []
        confidence = []
        for i in range(0, len(keypoints), 3):
            keypoints_x.append(keypoints[i])
            keypoints_y.append(keypoints[i + 1])
            confidence.append(keypoints[i + 2])
        ax.scatter(keypoints_x, keypoints_y)
        for i in BK.list_link.value:
            ax.plot([keypoints_x[i[0]], keypoints_x[i[1]]],
                    [keypoints_y[i[0]], keypoints_y[i[1]]])
    if show_frame:
        img = np.asarray(Image.open(path_frame))
        plt.imshow(img)
    else:
        ax.set_ylim(ax.get_ylim()[::-1])  # (0,0) in top left hand corner
    plt.show()


def _skel_width_height(keypoints: list):
    """Used by noise_2d(): Returns the max width and height of
       skeletons keypoints
    """
    xmax = max((val, index) for index, val in enumerate(keypoints) if index % 3 == 0)[0]
    xmin = min((val, index) for index, val in enumerate(keypoints) if index % 3 == 0)[0]
    ymax = min((val, index) for index, val in enumerate(keypoints) if index % 3 == 1)[0]
    ymin = max((val, index) for index, val in enumerate(keypoints) if index % 3 == 1)[0]
    xscale = (int(xmax) - int(xmin)) / 100
    yscale = (int(ymax) - int(ymin)) / 100
    return xscale, yscale


def noise_2d(input_folder_path: Path,
             json_name: list[str],
             output_folder_path: Path,
             num_copy: int = 3,
             noise_magnitude: float = 4.0):
    """
    Generate 1 json per number of copy asked + the original one. Add
    randomly add/remove random noise of 1% of the skeleton maximum
    amplitude to each keypoints coordinates, this 1% max value is
    multiplied by noise_magnitude

    Parameters
    ----------
    input_folder_path : Path,
                path of the folder where the original json are located
    json_name : list[str],
                a list of json files names that are going to be updated
                Ex: ["file.json"]
    output_folder_path : Path,
                  path of the folder where the new generated json are
                  saved
    num_copy : int,
               number of copy to make with slight rotation until it
               reaches max_angle
    noise_magnitude : float,
                      coefficient the random noise of maximum 1% of
                      total skeleton amplitude is multiplied by
    """
    new_json_name = []
    for file_name in json_name:
        with open(str(input_folder_path / file_name)) as file:
            data = json.load(file)
            num_frame = len(data['frames'])
        new_json_name.append(file_name)
        for copy in range(num_copy):
            noisy_data = data
            for frame in range(0, num_frame):
                for skeleton in range(len(noisy_data['frames'][frame]['skeletons'])):
                    xscale, yscale = _skel_width_height(noisy_data['frames'][frame]['skeletons'][skeleton]["keypoints"])
                    noise_for_facex = noise_magnitude * random.random() * xscale * random.choice([-1, 1])
                    noise_for_facey = noise_magnitude * random.random() * yscale * random.choice([-1, 1])
                    for point in range(0, 5 * 3, 3):  # uniform noise for the face
                        noisy_data['frames'][frame]['skeletons'][skeleton]['keypoints'][point] += noise_for_facex
                        noisy_data['frames'][frame]['skeletons'][skeleton]['keypoints'][point + 1] += noise_for_facey
                    for point in range(5 * 3, len(noisy_data['frames'][frame]['skeletons'][skeleton]['keypoints']), 3):
                        # not changing face keypoints
                        noisy_data['frames'][frame]['skeletons'][skeleton]['keypoints'][point] += (
                                noise_magnitude * random.random() * xscale * random.choice([-1, 1]))
                        noisy_data['frames'][frame]['skeletons'][skeleton]['keypoints'][point + 1] += (
                                noise_magnitude * random.random() * yscale * random.choice([-1, 1]))
            new_json_name.append(file_name.strip(".json") + "_N" + str(copy) + ".json")
            with open(str(output_folder_path / new_json_name[len(new_json_name) - 1]),
                      'w') as outfile:
                json.dump(noisy_data, outfile)
    return new_json_name


def _get_transform_matrix(resolution: list[int, int],
                          translation: list,
                          rotation: list,
                          scaling: list):
    """Create the transform matrix using cartesian dimension"""
    # split input
    t_x, t_y, t_z = translation
    r_x, r_y, r_z = rotation
    s_x, s_y, s_z = scaling
    # degrees to rad
    theta_rx = np.deg2rad(r_x)
    theta_ry = np.deg2rad(r_y)
    theta_rz = np.deg2rad(r_z)
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


def _cart_augment(M,
                  original_keypoints: list):
    """
    Compute the new keypoints value using the cartesian dimension
    Matrix
    """
    keypoints = []
    for i in range(0, len(original_keypoints) - 1, 3):
        keypoints.append([original_keypoints[i], original_keypoints[i + 1]])
    keypoints = np.array([keypoints], dtype="float")
    keypoints = cv2.perspectiveTransform(keypoints, M)
    for i in range(0, len(keypoints[0])):
        original_keypoints[3 * i] = keypoints[0][i][0].tolist()
        original_keypoints[3 * i + 1] = keypoints[0][i][1].tolist()
    return original_keypoints


def multiaugment(input_folder_path: Path,
                 json_name: list[str],
                 output_folder_path: Path,
                 translation: list,
                 rotation: list,
                 scaling: list):
    """
    input_folder_path : Path,
                path of the folder where the original json are located
    json_name : list[str],
                a list of json files names that are going to be updated
                Ex: ["file.json"]
    output_folder_path : Path,
                  path of the folder where the new generated json are
                  saved
    translation : list,
                  list of the parameters for translation matrix
                  [tx, ty, tz]
    rotation : list,
               list of the parameters for rotation matrix [rx, ry, rz]
    scaling : list,
              list of the parameters for the scaling matrix
              [sx, sy, sz]
    """
    new_json_name = []
    for file_name in json_name:
        with open(str(input_folder_path / file_name)) as file:
            data = json.load(file)
            resolution = data["resolution"]
            num_frame = len(data['frames'])
        new_data = data
        for frame in range(0, num_frame):
            for skeleton in range(len(new_data['frames'][frame]['skeletons'])):
                new_data['frames'][frame]['skeletons'][skeleton]["keypoints"] = (
                    _cart_augment(_get_transform_matrix(resolution, translation, rotation, scaling),
                                  new_data['frames'][frame]['skeletons'][skeleton]["keypoints"]))
        new_json_name.append(file_name)
        with open(str(output_folder_path / file_name),
                  'w') as outfile:
            json.dump(new_data, outfile)
    return new_json_name


def grid_augment(path_json: Path,
                 grid: gridParam,
                 max_copy: int = -1):
    """
    Generate multiple json from an original json with different types
    of augments like translation, rotation, scaling on all 3 axis.

    Parameters
    ----------
    path_json : Path,
                path where the original json file is located
    grid : gridParam,
           storing all needed parameters for augments
    max_copy : int,
               Maximum copy of an original file that can be
               generated
    """
    parent_folder = path_json.parent
    choice_rotation = list(product(grid.rotation[0], grid.rotation[1], grid.rotation[2]))
    choice_translation = list(product(grid.translation[0], grid.translation[1], grid.translation[2]))
    choice_scaling = grid.scaling
    choice_noise = list(product(grid.noise[0], grid.noise[1]))
    generator = product(choice_translation, choice_rotation, choice_scaling, choice_noise)
    next(generator)  # don't look at original value
    generated_pic = 0
    counter = 1
    if max_copy == -1:
        flag_limit = False
    else:
        flag_limit = True
    with open(path_json) as file:
        original_data = json.load(file)
    for indices in generator:
        if flag_limit and generated_pic >= max_copy:
            print("Max copy overflow")
            break
        else:
            # create copy json with final name
            new_name = path_json.stem + "_augment_" + str(counter) + ".json"
            with open(str(parent_folder / new_name),
                      'w') as outfile:
                json.dump(original_data, outfile)
            result_multiaugment = multiaugment(parent_folder, [new_name], parent_folder, indices[0], indices[1],
                                               indices[2])
            result_noise = noise_2d(parent_folder, result_multiaugment, parent_folder, indices[3][0], indices[3][1])
            generated_pic += len(result_noise)
            counter += 1
    return generated_pic
