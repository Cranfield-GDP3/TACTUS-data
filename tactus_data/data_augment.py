import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
from enum import Enum


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

def plot_skeleton2D(path_json : Path,
                    path_frame : Path):
    """
    Plot the 2D skeleton on top of the corresponding frame for testing purpose on the different augment

    Parameters
    ----------
    path_json : Path, path where the json file is located
    path_frame : Path, path where the frame file is located


    Example of Path input : "C:/Users/alcharyx/TACTUS-data/data/ut-interaction"
    """

    file = open(str(path_json))
    data = json.load(file)

    keypoints = data['frames'][0]['skeletons'][0]['keypoints']
    keypoints_x = []
    keypoints_y = []
    confidence = []
    for i in range(0, len(keypoints), 3):
        keypoints_x.append(keypoints[i])
        keypoints_y.append(keypoints[i + 1])
        confidence.append(keypoints[i + 2])
    img = np.asarray(Image.open(str(path_frame)))
    plt.imshow(img)
    plt.scatter(keypoints_x, keypoints_y)
    plt.show()


def D2_flip_H(path_file : Path,
              path_output : Path):
    """
    Duplicate skeleton json data and flip horizontally one of the file.

    Parameters
    ----------
    path_file : Path, path where the original json files are located
    path_output : Path, path where the new generated data are saved


    Example of Path input : "C:/Users/alcharyx/TACTUS-data/data/ut-interaction"
    """
    list_files = Path.iterdir(path_file)
    # Loop into all the json file we want to augment
    for file_path in list_files:

        file = open(str(file_path))
        data = json.load(file)
        file_name = file_path.name

        # Get resolution + frames data
        shape = data['resolution']
        file.close()
        num_frame = len(data['frames'])
        flip_data = data

        # j loop to go into all frames per video
        for j in range(0, num_frame):
            # k loop to go into all skeletons per frame
            for k in range(len(flip_data['frames'][j]['skeletons'])):
                # l loop to go into all keypoints per skeleton, 17 keypoints each composed of x,y,confidence
                for l in range(0,len(flip_data['frames'][j]['skeletons'][k]['keypoints']),3):
                    flip_data['frames'][j]['skeletons'][k]['keypoints'][l] = shape[0] - flip_data['frames'][j]['skeletons'][k]['keypoints'][l]

        # Generate flipped json in output folder
        with open(str(path_output) + "\\" + str(file_name).strip(".json") + "_H_flip.json", 'w') as outfile:
            json.dump(flip_data,outfile)
        # Generate original json in output folder
        with open(str(path_output) +"\\" + str(file_name), 'w') as outfile:
            json.dump(flip_data,outfile)

        # for more readeable json use :
        # json_object = json.dumps(flipped_data,indent=4)
        # with open(str(path_output) + str(file_name).strip(".json") + "_H_flip.json", 'w') as outfile:
        #   outfile.write(json_object)

def _Rotate_center(keypoints : list,
                  angle : float,
                  resolution : list,
                  center_of_rotation : tuple
                  ):
    """
    Generate 1 json per number of copy asked + the original one. Each copy is rotated more and more until it reaches the
    max_angle. You can pick where you want the center of rotation of the skeleton to be

    Parameters
    ----------
    keypoints : list, all the 17 keypoints coordinates x,y,confidence (total of 51 values)
    angle : float, value of the rotation angle in radian
    resolution
    center_of_rotation : tuple, allow to compute the center or rotation for the skeleton, use BodyKeypoints
    class as reference, will do the center value among all the keypoint coordinates in the tuple
    """

    # Compute skeleton center of rotation depending on center_of_rotation parameter
    centerx=0
    centery=0
    for i in center_of_rotation:
        centerx = centerx + keypoints[i.value * 3 ]
        centery = centery + keypoints[i.value * 3 +1]

    centerx = centerx / len(center_of_rotation)
    centery = centery / len(center_of_rotation)

    # Rotation around (0,0) coordinates
    center_pic_x = 0
    center_pic_y = 0

    #See change needed to center skeleton on (0,0)
    diffx =center_pic_x - centerx
    diffy = center_pic_y - centery

    centered_keypoints = [] # To store skeleton centered on (0,0)
    rotated_keypoints =[] # To store skeleton centered and rotated of angle
    decenter_keypoints = [] # To store rotated skeleton back the original center
    for i in range(0,len(keypoints),3):

        # Center
        centered_keypoints.append(keypoints[i]+diffx)
        centered_keypoints.append(keypoints[i+1]+diffy)
        centered_keypoints.append(keypoints[i+2])

        # Rotate
        # rotation_matrix 2D = [[np.cos(i),-np.sin(i)],
        #                      [np.sin(i),np.cos(i)]]

        rotated_keypoints.append(centered_keypoints[i] * np.cos(angle) - centered_keypoints[i+1] * np.sin(angle))
        rotated_keypoints.append(centered_keypoints[i] * np.sin(angle) + centered_keypoints[i+1] * np.cos(angle))
        rotated_keypoints.append(centered_keypoints[i+2])

        # Decenter
        decenter_keypoints.append(rotated_keypoints[i] - diffx)
        decenter_keypoints.append(rotated_keypoints[i+1]-diffy)
        decenter_keypoints.append(rotated_keypoints[i+2])

    return decenter_keypoints



def D2_Rotation(path_file : Path,
                path_output : Path,
                max_angle: float = 10.0,
                num_copy : int = 3,
                rotate_center : tuple = (BodyKeypoints.LAnkle, BodyKeypoints.RAnkle)):
    """
    Generate 1 json per number of copy asked + the original one. Each copy is rotated more and more until it reaches the
    max_angle. You can pick where you want the center of rotation of the skeleton to be

    Parameters
    ----------
    path_file : Path, path where the original json files are located
    path_output : Path, path where the new generated data are saved
    max_angle : float, value of the max rotation angle in degree
    num_copy : int, number of copy to make with slight rotation until it reaches max_angle
    rotate_center : tuple, allow to compute the center or rotation for the skeleton, use BodyKeypoints
    class as reference, will do the center value among all the keypoint coordinates in the tuple

    Example of Path input : "C:/Users/alcharyx/TACTUS-data/data/ut-interaction"
    """

    list_files = Path.iterdir(path_file)
    rad_angle = np.radians(max_angle)
    list_angle = np.linspace(0,rad_angle,num_copy+1) #start 0 to keep original

    # Loop into all the json file we want to augment
    for file_path in list_files:
        # Extract json data
        file = open(str(file_path))
        data = json.load(file)
        file_name = file_path.name
        file.close()

        shape = data['resolution']
        num_frame = len(data['frames'])
        # i loop to copy with all the different angles
        for i in range(len(list_angle)):
            # Local copy of original json
            rotated_data = data
            # j loop to go into all frames per video
            for j in range(0, num_frame):
                # k loop to go into all skeletons per frame
                for k in range(len(rotated_data['frames'][j]['skeletons'])): # Look for upgrade with enumerate
                    rotated_data['frames'][j]['skeletons'][k]['keypoints'] = \
                        _Rotate_center(keypoints=rotated_data['frames'][j]['skeletons'][k]['keypoints'],
                                      angle=list_angle[i], resolution=shape, center_of_rotation= rotate_center)

            # Generate rotated json in output folder
            with open(str(path_output) + "\\" + str(file_name).strip(".json") + "_Rotated" + str(i) + ".json", 'w') as outfile:
                json.dump(rotated_data, outfile)

# Test functions
#D2_Rotation(Path("D:/Documents/Cranfield/GDP/TACTUS-data/data/test_skeleton/original"),Path("D:/Documents/Cranfield/GDP/TACTUS-data/data/test_skeleton/Rotation"))
#plot_skeleton2D(Path("D:/Documents/Cranfield/GDP/TACTUS-data/data/test_skeleton/Rotation/test_2_skeleton_051_Rotated3.json"),Path("D:/Documents/Cranfield/GDP/TACTUS-data/data/test_skeleton/051.jpg"))

