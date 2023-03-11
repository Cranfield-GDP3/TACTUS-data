import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
from enum import Enum




def plot_skeleton2D(path_json : Path,
                    path_frame : Path
                    ):
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

    Working directory is expected to be **/TACTUS-data
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


