import json
import random
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


class BodyKeypoints(Enum):
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


def _check_on_frame(keypoints: list,
                    resolution: list) -> bool:
    """
    Plot the 2D skeleton on top of the corresponding frame for testing purpose on the different augment

    Parameters
    ----------
    keypoints : list,
                list of the new generated keypoints to check
    resolution : list,
                 list of [x,y] resolution of the original frame
        """
    flag_ok = True
    for i in range(0, len(keypoints), 3):
        if keypoints[i] > resolution[0] or keypoints[i] < 0 or keypoints[i+1] > resolution[1] or keypoints[i+1] < 0:
            flag_ok = False
    return flag_ok


def plot_skeleton_2d(path_json: Path,
                     path_frame: Path):
    """
    Plot the 2D skeleton on top of the corresponding frame for testing purpose on the different augment

    Parameters
    ----------
    path_json : Path,
                path where the json file is located
    path_frame : Path,
                 path where the frame file is located
    """
    with open(path_json) as file:
        data = json.load(file)
    keypoints = data['frames'][0]['skeletons'][0]['keypoints']
    keypoints_x = []
    keypoints_y = []
    confidence = []
    for i in range(0, len(keypoints), 3):
        keypoints_x.append(keypoints[i])
        keypoints_y.append(keypoints[i + 1])
        confidence.append(keypoints[i + 2])
    img = np.asarray(Image.open(path_frame))
    plt.imshow(img)
    plt.scatter(keypoints_x, keypoints_y)
    plt.show()


def flip_h_2d(path_file: Path,
              path_output: Path):
    """
    Duplicate skeleton json data and flip horizontally one of the file.

    Parameters
    ----------
    path_file : Path,
                path where the original json files are located
    path_output : Path,
                  path where the new generated data are saved
    """
    list_files = Path.iterdir(path_file)
    augment_ok = True
    for file_path in list_files:
        file_name = file_path.name
        with open(file_path) as file:
            data = json.load(file)
        shape = data['resolution']
        num_frame = len(data['frames'])
        flip_data = data
        for frame in range(0, num_frame):
            for skeleton in range(len(flip_data['frames'][frame]['skeletons'])):
                for point in range(0, len(flip_data['frames'][frame]['skeletons'][skeleton]['keypoints']), 3):
                    flip_data['frames'][frame]['skeletons'][skeleton]['keypoints'][point] = (
                        shape[0] - flip_data['frames'][frame]['skeletons'][skeleton]['keypoints'][point])
        if augment_ok:
            with open(str(path_output) + "\\" + str(file_name).strip(".json") + "_H_flip.json", 'w') as outfile:
                json.dump(flip_data, outfile)


def _rotate_center(keypoints: list,
                   angle: float,
                   center_of_rotation: tuple
                   ):
    """
    Generate 1 json per number of copy asked + the original one. Each copy is rotated more and more until it reaches the
    max_angle. You can pick where you want the center of rotation of the skeleton to be

    Parameters
    ----------
    keypoints : list,
                all the 17 keypoints coordinates x,y,confidence (total of 51 values)
    angle : float,
            value of the rotation angle in radian
    center_of_rotation : tuple, allow to compute the center or rotation for the skeleton, use BodyKeypoints
    class as reference, will do the center value among all the keypoint coordinates in the tuple
    """

    # Compute skeleton center of rotation depending on center_of_rotation parameter
    centerx = 0
    centery = 0
    for i in center_of_rotation:
        centerx = centerx + keypoints[i.value * 3]
        centery = centery + keypoints[i.value * 3 + 1]
    centerx = centerx / len(center_of_rotation)
    centery = centery / len(center_of_rotation)
    # Rotation around (0,0) coordinates
    center_pic_x = 0
    center_pic_y = 0
    diffx = center_pic_x - centerx
    diffy = center_pic_y - centery
    centered_keypoints = []
    rotated_keypoints = []
    decenter_keypoints = []
    for i in range(0, len(keypoints), 3):
        centered_keypoints.append(keypoints[i] + diffx)
        centered_keypoints.append(keypoints[i + 1] + diffy)
        centered_keypoints.append(keypoints[i + 2])
        # rotation_matrix 2D = [[np.cos(i),-np.sin(i)],
        #                      [np.sin(i),np.cos(i)]]
        rotated_keypoints.append(centered_keypoints[i] * np.cos(angle) - centered_keypoints[i + 1] * np.sin(angle))
        rotated_keypoints.append(centered_keypoints[i] * np.sin(angle) + centered_keypoints[i + 1] * np.cos(angle))
        rotated_keypoints.append(centered_keypoints[i + 2])
        decenter_keypoints.append(rotated_keypoints[i] - diffx)
        decenter_keypoints.append(rotated_keypoints[i + 1] - diffy)
        decenter_keypoints.append(rotated_keypoints[i + 2])
    return decenter_keypoints


def rotation_2d(path_file: Path,
                path_output: Path,
                max_angle: float = 10.0,
                num_copy: int = 3,
                rotate_center: tuple = (BodyKeypoints.LAnkle, BodyKeypoints.RAnkle)):
    """
    Generate 1 json per number of copy asked + the original one. Each copy is rotated more and more until it reaches the
    max_angle. You can pick where you want the center of rotation of the skeleton to be

    Parameters
    ----------
    path_file : Path,
                path where the original json files are located
    path_output : Path,
                  path where the new generated data are saved
    max_angle : float,
                value of the max rotation angle in degree
    num_copy : int,
               number of copy to make with slight rotation until it reaches max_angle
    rotate_center : tuple,
                    allow to compute the center or rotation for the skeleton, use BodyKeypoints class as reference,
                    will do the center value among all the keypoint coordinates in the tuple
    """

    list_files = Path.iterdir(path_file)
    augment_ok = True
    rad_angle = np.radians(max_angle)
    list_angle = np.linspace(0, rad_angle, num_copy + 1)  # start 0 to keep original
    for file_path in list_files:
        file_name = file_path.name
        with open(file_path) as file:
            data = json.load(file)
            resolution = data["resolution"]
        num_frame = len(data['frames'])
        for angl in range(1,len(list_angle)):
            rotated_data = data
            for frame in range(0, num_frame):
                for skeleton in range(len(rotated_data['frames'][frame]['skeletons'])):
                    rotated_data['frames'][frame]['skeletons'][skeleton]['keypoints'] = (
                        _rotate_center(keypoints=rotated_data['frames'][frame]['skeletons'][skeleton]['keypoints'],
                                       angle=list_angle[angl], center_of_rotation=rotate_center))
                    if not _check_on_frame(rotated_data['frames'][frame]['skeletons'][skeleton]["keypoints"],
                                           resolution):
                        augment_ok = False
            if augment_ok:
                with open(str(path_output) + "\\" + str(file_name).strip(".json") + "_Rotated" + str(angl) + ".json",
                          'w') as outfile:
                    json.dump(rotated_data, outfile)


def noise_2d(path_file: Path,
             path_output: Path,
             num_copy: int = 3,
             noise_magnitude: float = 4.0):
    """
    Generate 1 json per number of copy asked + the original one. Add randomly add/remove random noise to each keypoints
    coordinates depending on noise_magnitude parameter

    Parameters
    ----------
    path_file : Path,
                path where the original json files are located
    path_output : Path,
                  path where the new generated data are saved
    num_copy : int,
               number of copy to make with slight rotation until it reaches max_angle
    noise_magnitude : float,
                      coefficient the random noise between 0 and 1 is multiplied by
    """
    list_files = Path.iterdir(path_file)
    augment_ok = True
    for file_path in list_files:
        file_name = file_path.name
        with open(file_path) as file:
            data = json.load(file)
            resolution = data["resolution"]
        num_frame = len(data['frames'])
        for copy in range(num_copy):
            noisy_data = data
            for frame in range(0, num_frame):
                for skeleton in range(len(noisy_data['frames'][frame]['skeletons'])):
                    for point in range(5, len(noisy_data['frames'][frame]['skeletons'][skeleton]['keypoints'])):
                        # not changing face keypoints
                        noisy_data['frames'][frame]['skeletons'][skeleton]['keypoints'][point] = (
                                noisy_data['frames'][frame]['skeletons'][skeleton]['keypoints'][point] +
                                noise_magnitude * random.random() * random.choice([-1, 1]))
                    if not _check_on_frame(noisy_data['frames'][frame]['skeletons'][skeleton]["keypoints"],
                                           resolution):
                        augment_ok = False
            if augment_ok:
                with open(str(path_output) + "\\" + str(file_name).strip(".json") + "_noise" + str(copy) + ".json",
                          'w') as outfile:
                    json.dump(noisy_data, outfile)


def _center_after_scaling(keypoints: list,
                          scale_keypoints: list,
                          resolution: list,
                          factor: float):
    res_center = [x / 2 for x in resolution]
    d_center = [keypoints[BodyKeypoints["Nose"].value] - res_center[0],
            keypoints[BodyKeypoints["Nose"].value + 1] - res_center[1]]
    new_d_center = [x * factor for x in d_center]
    desired = [res_center[0]+new_d_center[0],res_center[1]+new_d_center[1]]
    diff = [scale_keypoints[BodyKeypoints["Nose"].value] -desired[0],
            scale_keypoints[BodyKeypoints["Nose"].value + 1] - desired[1]]
    for i in range(0, len(keypoints) - 1, 3):
        scale_keypoints[i] = scale_keypoints[i] - diff[0]
        scale_keypoints[i + 1] = scale_keypoints[i + 1] - diff[1]
    return scale_keypoints

def _uniform_scale(keypoints: list,
                   distance_change: float,
                   focal_length: float,
                   resolution: list):
    # 1/3" Type image sensor with 4:3 aspect ratio 4.8mm H * 3.6 mm V
    sensor1_3 = [4.8, 3.6]
    # Distance is arbitrarly 10 meters
    # Scale
    distance = 10
    diagonal = np.sqrt(sensor1_3[0]*sensor1_3[0] + sensor1_3[1]*sensor1_3[1])
    fov = 2 * np.arctan(diagonal / (2 * focal_length))
    new_focal_length = focal_length * (distance+distance_change)/distance
    new_fov = 2 * np.arctan(diagonal / (2 * new_focal_length))
    factor = new_fov / fov
    scale_keypoints = []
    for i in range(0, len(keypoints)-1, 3):
        scale_keypoints.append(keypoints[i]*factor)
        scale_keypoints.append(keypoints[i+1] * factor)
        scale_keypoints.append(keypoints[i+2])
    scale_keypoints = _center_after_scaling(keypoints, scale_keypoints, resolution, factor)
    return scale_keypoints


def camera_distance_2d(path_file: Path,
                       path_output: Path,
                       distance: float,
                       focal_length: float = 3.6):
    """
    Generate 1 json with a new scaling of skeletons. The distance parameter allows you to virtually move the camera
    further or closer to the frame so that the scale change accordingly, distance is in meters represent the distance
    added on the original position of the camera, if the camera goes closer(-) / further(+).

    Parameters
    ----------
    path_file : Path,
                path where the original json files are located
    path_output : Path,
                    path where the new generated data are saved
    distance : float,
               change the camera distance by a positive or negative number of meter to the scene
               negative means closer positive means further. Don't put -10 as value since it will mean the camera is
               inside of the picture (arbitrarly put at 10 meters)
    focal_length : float,
                   The focal length of the camera in millimetres, it impacts the angle of view of the camera and
                   will influence the change of scale compare to the distance. Here is a usual CCTV focal length
                   with the angle of view of the camera:
                    Focal Lenght / Angle of View / Clear view
                    -   2.8 mm   /      108°     /    5 m
                    -   3.6 mm   /      82.6°    /    8 m
                    -   4.0 mm   /      38°      /   12 m
                    -   6.0 mm   /      54°      /   18 m
        """
    list_files = Path.iterdir(path_file)
    augment_ok = True
    for file_path in list_files:
        file_name = file_path.name
        with open(file_path) as file:
            data = json.load(file)
            resolution = data["resolution"]
        num_frame = len(data['frames'])
        scaled_data = data
        for frame in range(0, num_frame):
            for skeleton in range(len(scaled_data['frames'][frame]['skeletons'])):
                scaled_data['frames'][frame]['skeletons'][skeleton]["keypoints"] = (
                    _uniform_scale(scaled_data['frames'][frame]['skeletons'][skeleton]["keypoints"],
                                   distance, focal_length, resolution))
                if not _check_on_frame(scaled_data['frames'][frame]['skeletons'][skeleton]["keypoints"], resolution):
                    augment_ok = False
        if augment_ok:
            with open(str(path_output) + "\\" + str(file_name).strip(".json") + "_scale" + str(distance) + ".json",
                      'w') as outfile:
                json.dump(scaled_data, outfile)
        else:
            print(str(file_name).strip(".json") + "_scale" + str(distance) + " is out of frame")


def anthropomorphic_scale(factor: float,
                          keypoints: list):
    factor = 1


def skeletons_scale():
    """
    Generate 1 json with new skeleton sizes, it is possible to change size of multiple skeleton on a picture
    with different factors the goal is to create interaction with different size individual

    Parameters
    ----------
    path_file : Path,
                path where the original json files are located
    path_output : Path,
                  path where the new generated data are saved
    max_factor : float,
                 Max value of the size change factor
    max_diff_factor : float,
                      Maximum difference between the factor of all skeletons
    """
