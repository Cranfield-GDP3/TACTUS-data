from pathlib import Path
from enum import Enum
from typing import Union
import cv2
from tactus_yolov7 import Yolov7

MODEL_WEIGHTS_PATH = Path("data/raw/model/yolov7-w6-pose.pt")


def yolov7(input_dir: Path, model: Yolov7):
    """extract skeleton keypoints with yolov7. It extract skeletons
    from every *.jpg image in the input_dir"""
    formatted_json = {}
    formatted_json["frames"] = []

    min_nbr_skeletons = float("inf")
    max_nbr_skeletons = 0
    for frame_path in Path(input_dir).glob("*.jpg"):
        frame_json = {"frame_id": frame_path.name}

        img = cv2.imread(str(frame_path))
        skeletons = model.predict_frame(img)

        min_nbr_skeletons = min(min_nbr_skeletons, len(skeletons))
        max_nbr_skeletons = max(max_nbr_skeletons, len(skeletons))

        for i, skeleton in enumerate(skeletons):
            skeletons[i]["keypoints"] = round_skeleton_kpts(skeleton["keypoints"], 3)

        frame_json["skeletons"] = skeletons
        formatted_json["frames"].append(frame_json)

    formatted_json["min_nbr_skeletons"] = min_nbr_skeletons
    formatted_json["max_nbr_skeletons"] = max_nbr_skeletons

    return formatted_json


def delete_confidence_kpt(skeleton: list) -> list:
    """delete the confidence for each keypoint of a skeleton"""
    return drop_every_nth_index(skeleton, 3)


def drop_every_nth_index(initial_list: list, n: int) -> list:
    """remove every nth index from a list"""
    del initial_list[n-1::n]
    return initial_list


def round_values(skeleton: list) -> list:
    """round all the values of a list. Useful to save a lot of space
    when saving the skeletons to a file"""
    return [round(kpt) for kpt in skeleton]


def round_skeleton_kpts(skeleton: list, n: int) -> list:
    """round all the values of a list except every nth index. Useful
    to save a lot of space when saving the skeletons to a file"""
    for i, kpt in enumerate(skeleton):
        if i%n != n-1: # kpt coordinates
            skeleton[i] = round(kpt)
        else: # kpt confidence
            skeleton[i] = round(kpt, 2)

    return skeleton


def keypoints_to_xy(keypoints: Union[list, tuple]) -> tuple[list, list]:
    """
    return a tuple of x and y coordinates

    Parameters
    ----------
    keypoints : Union[list, tuple]
        keypoints of a skeleton. Can either be a long list of
        bodykeypoints like [x1, y1, c1, x2, y2, c2, ...] or a tuple of
        x, y coordinates ([x1, x2, ...], [y1, y2, ...])

    Returns
    -------
    tuple[list, list]
        tuple of x, y coordinates ([x1, x2, ...], [y1, y2, ...])
    """
    if isinstance(keypoints[0], (int, float)):
        return keypoints[::3], keypoints[1::3]

    return keypoints


def skeleton_bbx(keypoints: Union[list, tuple]) -> tuple[int, int, int, int]:
    """
    return the x_left, y_bottom, width and height of a skeleton

    Parameters
    ----------
    keypoints : Union[list, tuple]
        keypoints of a skeleton. Can either be a long list of
        bodykeypoints like [x1, y1, c1, x2, y2, c2, ...] or a tuple of
        x, y coordinates ([x1, x2, ...], [y1, y2, ...])

    Returns
    -------
    tuple[int, int, int, int]
        (x_left, y_bottom, width, height)
    """
    keypoints_x, keypoints_y = keypoints_to_xy(keypoints)

    min_x = min(keypoints_x)
    max_x = max(keypoints_x)
    min_y = min(keypoints_y)
    max_y = max(keypoints_y)

    return min_x, min_y, max_x-min_x, max_y-min_y


class BK(Enum):
    """represent a skeleton body keypoints"""
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
