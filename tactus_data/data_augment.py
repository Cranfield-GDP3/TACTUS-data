import json
import random
import time
from itertools import product
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


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
            # print("Value ",keypoints[i],"    Resolution :",resolution)
            flag_ok = False
            return flag_ok
    return flag_ok


def plot_skeleton_2d(path_json: Path,
                     path_frame: Path,
                     show_frame: bool = True):
    """
    Plot the 2D skeleton (keypoint and limbs) on top of the corresponding frame for testing purpose on the different augment

    Parameters
    ----------
    path_json : Path,
                path where the json file is located
    path_frame : Path,
                 path where the frame file is located
    show_frame : bool,
                 If false allows you to see only skeleton
    """
    with open(path_json) as file:
        data = json.load(file)
    fig, ax = plt.subplots()
    for skeletons in data['frames'][20]['skeletons']:
        keypoints = skeletons["keypoints"]
        keypoints_x = []
        keypoints_y = []
        confidence = []
        for i in range(0, len(keypoints), 3):
            keypoints_x.append(keypoints[i])
            keypoints_y.append(keypoints[i + 1])
            confidence.append(keypoints[i + 2])

        list_link =[(BK.RAnkle,BK.RKnee),
                    (BK.LAnkle, BK.LKnee),
                    (BK.RKnee, BK.RHip),
                    (BK.LKnee, BK.LHip),
                    (BK.RHip,BK.LHip),
                    (BK.RHip,BK.RShoulder),
                    (BK.LHip,BK.LShoulder),
                    (BK.RShoulder,BK.LShoulder),
                    (BK.RShoulder,BK.RElbow),
                    (BK.RElbow,BK.RWrist),
                    (BK.LShoulder,BK.LElbow),
                    (BK.LElbow,BK.LWrist),
                    (BK.Nose,BK.LEye),
                    (BK.Nose,BK.REye),
                    (BK.LEye,BK.LEar),
                    (BK.REye,BK.REar)]
        ax.scatter(keypoints_x, keypoints_y)
        for i in list_link:
            ax.plot([keypoints_x[i[0].value],keypoints_x[i[1].value]],[keypoints_y[i[0].value],keypoints_y[i[1].value]])


    if show_frame == True:
        img = np.asarray(Image.open(path_frame))
        plt.imshow(img)
    else :
        ax.set_ylim(ax.get_ylim()[::-1])  # (0,0) in top left hand corner
    plt.show()


def flip_h_2d(input_folder_path: Path,
              json_name: list[str],
              output_folder_path: Path):
    """
    Duplicate skeleton json data and flip horizontally one of the file.

    Parameters
    ----------
    input_folder_path : Path,
                path of the folder where the original json are located
    json_name : list[str],
                a list of json files names that are going to be updated Ex: ["file.json"]
    output_folder_path : Path,
                  path of the folder where the new generated json are saved
    """
    augment_ok = True
    for file_name in json_name:
        with open(str(input_folder_path) + "\\" + file_name) as file:
            data = json.load(file)
            resolution = data['resolution']
            num_frame = len(data['frames'])
        flip_data = data
        for frame in range(0, num_frame):
            for skeleton in range(len(flip_data['frames'][frame]['skeletons'])):
                for point in range(0, len(flip_data['frames'][frame]['skeletons'][skeleton]['keypoints']), 3):
                    flip_data['frames'][frame]['skeletons'][skeleton]['keypoints'][point] = (
                        resolution[0] - flip_data['frames'][frame]['skeletons'][skeleton]['keypoints'][point])
                if not _check_on_frame(flip_data['frames'][frame]['skeletons'][skeleton]["keypoints"],
                                       resolution):
                    augment_ok = True
        if augment_ok:
            with open(str(output_folder_path) + "\\" + str(file_name), 'w') as outfile:
                json.dump(flip_data, outfile)
        else:
            print(str(file_name) + " is out of frame in flip augment")
    return json_name


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
    center_of_rotation : tuple, allow to compute the center or rotation for the skeleton, use BK
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


def rotation_2d(input_folder_path: Path,
                json_name: list[str],
                output_folder_path: Path,
                max_angle: float = 10.0,
                num_copy: int = 3,
                rotate_center: tuple = (BK.LAnkle, BK.RAnkle)):
    """
    Generate 1 json per number of copy asked + the original one. Each copy is rotated more and more until it reaches the
    max_angle. You can pick where you want the center of rotation of the skeleton to be

    Parameters
    ----------
    input_folder_path : Path,
                path of the folder where the original json are located
    json_name : list[str],
                a list of json files names that are going to be updated Ex: ["file.json"]
    output_folder_path : Path,
                  path of the folder where the new generated json are saved
    max_angle : float,
                value of the max rotation angle in degree
    num_copy : int,
               number of copy to make with slight rotation until it reaches max_angle
    rotate_center : tuple,
                    allow to compute the center or rotation for the skeleton, use BK class as reference,
                    will do the center value among all the keypoint coordinates in the tuple
    """
    augment_ok = True
    rad_angle = np.radians(max_angle)
    list_angle = np.linspace(0, rad_angle, num_copy + 1)
    new_json_name = []
    for file_name in json_name:
        with open(str(input_folder_path) + "\\" + file_name) as file:
            data = json.load(file)
            resolution = data["resolution"]
            num_frame = len(data['frames'])
        new_json_name.append(file_name)
        for angl in range(1, len(list_angle)):  # start 1 to not take the original
            rotated_data = data
            for frame in range(0, num_frame):
                for skeleton in range(len(rotated_data['frames'][frame]['skeletons'])):
                    rotated_data['frames'][frame]['skeletons'][skeleton]['keypoints'] = (
                        _rotate_center(keypoints=rotated_data['frames'][frame]['skeletons'][skeleton]['keypoints'],
                                       angle=list_angle[angl], center_of_rotation=rotate_center))
                    if not _check_on_frame(rotated_data['frames'][frame]['skeletons'][skeleton]["keypoints"],
                                           resolution):
                        augment_ok = True
            if augment_ok:
                new_json_name.append(file_name.strip(".json") + "_R" + str(angl) + ".json")
                with open(str(output_folder_path) + "\\" + new_json_name[len(new_json_name)-1],
                          'w') as outfile:
                    json.dump(rotated_data, outfile)
            else:
                print(new_json_name[len(new_json_name)-1] + " is out of frame in rotation augment")
                new_json_name = json_name
    return new_json_name


def _skel_width_height(keypoints: list):

    xmax = max((val, index) for index, val in enumerate(keypoints) if index % 3 == 0)[0]
    xmin = min((val, index) for index, val in enumerate(keypoints) if index % 3 == 0)[0]
    ymax = min((val, index) for index, val in enumerate(keypoints) if index % 3 == 1)[0]
    ymin = max((val, index) for index, val in enumerate(keypoints) if index % 3 == 1)[0]

    xscale = (int(xmax) - int(xmin)) / 100
    yscale = (int(ymax) - int(ymin)) / 100
    return xscale,yscale


def noise_2d(input_folder_path: Path,
             json_name: list[str],
             output_folder_path: Path,
             num_copy: int = 3,
             noise_magnitude: float = 4.0):
    """
    Generate 1 json per number of copy asked + the original one. Add randomly add/remove random noise of 1% of
    the skeleton maximum amplitude to each keypoints coordinates, this 1% max value is multiplied by noise_magnitude

    Parameters
    ----------
    input_folder_path : Path,
                path of the folder where the original json are located
    json_name : list[str],
                a list of json files names that are going to be updated Ex: ["file.json"]
    output_folder_path : Path,
                  path of the folder where the new generated json are saved
    num_copy : int,
               number of copy to make with slight rotation until it reaches max_angle
    noise_magnitude : float,
                      coefficient the random noise of maximum 1% of total skeleton amplitude is multiplied by
    """
    augment_ok = True
    new_json_name = []
    for file_name in json_name:
        with open(str(input_folder_path) + "\\" + file_name) as file:
            data = json.load(file)
            resolution = data["resolution"]
            num_frame = len(data['frames'])
        new_json_name.append(file_name)
        for copy in range(num_copy):
            noisy_data = data
            for frame in range(0, num_frame):
                for skeleton in range(len(noisy_data['frames'][frame]['skeletons'])):
                    xscale, yscale = _skel_width_height(noisy_data['frames'][frame]['skeletons'][skeleton]["keypoints"])
                    noise_for_facex = noise_magnitude * random.random() * xscale * random.choice([-1, 1])
                    noise_for_facey = noise_magnitude * random.random() * yscale * random.choice([-1, 1])
                    for point in range(0,5*3,3):  # uniform noise for the face
                        noisy_data['frames'][frame]['skeletons'][skeleton]['keypoints'][point] += noise_for_facex
                        noisy_data['frames'][frame]['skeletons'][skeleton]['keypoints'][point] += noise_for_facey
                    for point in range(5*3, len(noisy_data['frames'][frame]['skeletons'][skeleton]['keypoints']), 3):
                        # not changing face keypoints
                        noisy_data['frames'][frame]['skeletons'][skeleton]['keypoints'][point] += noise_magnitude * random.random() * xscale * random.choice([-1, 1])
                        noisy_data['frames'][frame]['skeletons'][skeleton]['keypoints'][point] += noise_magnitude * random.random() * yscale * random.choice([-1, 1])
                    if not _check_on_frame(noisy_data['frames'][frame]['skeletons'][skeleton]["keypoints"],
                                           resolution):
                        augment_ok = True
            if augment_ok:
                new_json_name.append(file_name.strip(".json") + "_N" + str(copy) + ".json")
                with open(str(output_folder_path) + "\\" + new_json_name[len(new_json_name)-1],
                          'w') as outfile:
                    json.dump(noisy_data, outfile)
            else:
                print(new_json_name[len(new_json_name)-1] + " is out of frame in noise augment")
                new_json_name = json_name
    return new_json_name


def _center_after_scaling(keypoints: list,
                          scale_keypoints: list,
                          resolution: list,
                          factor: float):
    res_center = [x / 2 for x in resolution]
    d_center = [keypoints[BK["Nose"].value] - res_center[0],
                keypoints[BK["Nose"].value + 1] - res_center[1]]
    new_d_center = [x * factor for x in d_center]
    desired = [res_center[0]+new_d_center[0], res_center[1]+new_d_center[1]]
    diff = [scale_keypoints[BK["Nose"].value] - desired[0],
            scale_keypoints[BK["Nose"].value + 1] - desired[1]]
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


def camera_distance_2d(input_folder_path: Path,
                       json_name: list[str],
                       output_folder_path: Path,
                       distance: float,
                       focal_length: float = 3.6):
    """
    Generate 1 json with a new scaling of skeletons. The distance parameter allows you to virtually move the camera
    further or closer to the frame so that the scale change accordingly, distance is in meters represent the distance
    added on the original position of the camera, if the camera goes closer(-) / further(+).

    Parameters
    ----------
    input_folder_path : Path,
                path of the folder where the original json are located
    json_name : list[str],
                a list of json files names that are going to be updated Ex: ["file.json"]
    output_folder_path : Path,
                  path of the folder where the new generated json are saved
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
    augment_ok = True
    new_json_name = []
    for file_name in json_name:
        with open(str(input_folder_path) + "\\" + file_name) as file:
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
                    augment_ok = True
        if augment_ok:
            new_json_name.append(file_name)
            with open(str(output_folder_path) + "\\" + str(file_name),
                      'w') as outfile:
                json.dump(scaled_data, outfile)
        else:
            print(str(file_name).strip(".json") + "_scale" + str(distance) + " is out of frame in scaling augment")
            new_json_name = json_name
    return new_json_name


def grid_augment(path_json: Path,
                 grid: list[list[list]],
                 max_copy: int = -1):
    """
    Generate 1 json with new skeleton sizes, it is possible to change size of multiple skeleton on a picture
    with different factors the goal is to create interaction with different size individual

    Parameters
    ----------
    path_json : Path,
                path where the original json file is located
    grid : list[list],
           the grid of value that will be use for the grid search the gris is a list of list, each element
           of the list a list of all the argument of each augment can take without including the path,
           in the form of another list :
           [list_flip_h, list_camera_distance_2d,list_rotation_2d,list_noise_2d]
           Ex:
           list_flip_h = [True, False]
           list_camera_distance_2d =[distance,focal_length]
           list_rotation_2d = [max_angle,num_copy,rotate_center]
           list_noise_2d = [num_copy, noise_magnitude]
           If you don't want an augment to be used just put an empty list
           list_camera_distance_2d :
            distance = [-5,-2, 2, 5, 10, 15] / focal_length = [2.8, 3.6, 4.0, 6.0]
    max_copy : int,
               Number of copy of the original file are going to be generated
    """
    parent_folder = path_json.parent
    list_flip_h = grid[0]
    list_camera_distance_2d = grid[1]
    list_rotation_2d = grid[2]
    list_noise_2d = grid[3]
    choice_flip = list(product(list_flip_h))
    choice_rot = list(product(list_rotation_2d[0], list_rotation_2d[1], list_rotation_2d[2]))
    choice_cam = list(product(list_camera_distance_2d[0], list_camera_distance_2d[1]))
    choice_noise = list(product(list_noise_2d[0], list_noise_2d[1]))
    generator = (product(choice_flip, choice_cam, choice_rot, choice_noise))
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
            break
        else:
            # create copy json with final name
            new_name = path_json.stem + "_augment_" + str(counter) + ".json"
            with open(str(parent_folder) + "\\" + str(new_name),
                      'w') as outfile:
                json.dump(original_data, outfile)
            # No parameter on flip so True / false
            if indices[0] == (True,):
                result_flip = flip_h_2d(parent_folder, [new_name], parent_folder)
            else:
                result_flip = [new_name]
            result_cam = camera_distance_2d(parent_folder, result_flip, parent_folder, indices[1][0], indices[1][1])
            result_rota = rotation_2d(parent_folder, result_cam, parent_folder, indices[2][0],
                                      indices[2][1], indices[2][2])
            result_noise = noise_2d(parent_folder, result_rota, parent_folder, indices[3][0], indices[3][1])
            generated_pic += len(result_noise)
            counter += 1
    if max_copy == -1:
        print("Generated :", generated_pic, " augmented copy of ", parent_folder.parts[len(parent_folder.parts)-2])
    else :
        print("Generation Maxed out ! Only generated ",generated_pic," augmented copy of ",parent_folder.parts[len(parent_folder.parts)-2])
    return  generated_pic



def augment_all_vid(input_folder_path: Path,
                    grid: list[list[list]],
                    fps: int,
                    max_copy: int = -1,
                    random_seed: int = 30000):
    random.seed(random_seed)
    total_cpy = 0
    t1 = time.time()
    list_dir = list(input_folder_path.iterdir())
    for index in range(len(list_dir)-2):
        vid_path = Path(str(list_dir[index]) + "\\" + str(fps) + "fps")
        vid_name = vid_path.glob('**/*.json')
        for json in vid_name:
            total_cpy += grid_augment(json,grid)
    t2 = time.time()
    time_total = (t2 - t1 ) / 60
    print("Increased data from ",len(list_dir)-1," to ",total_cpy,"in ", round(time_total,2)," minutes")






