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
            skeletons[i]["keypoints"] = king_of_france(skeletons[i]["keypoints"])
            skeletons[i]["keypoints"] = round_skeleton_kpts(skeleton["keypoints"])
            skeletons[i]["keypoints"] = remove_confidence_points(skeletons[i]["keypoints"])

        frame_json["skeletons"] = skeletons
        formatted_json["frames"].append(frame_json)

    formatted_json["min_nbr_skeletons"] = min_nbr_skeletons
    formatted_json["max_nbr_skeletons"] = max_nbr_skeletons
    formatted_json["resolution"] = img.shape[:2]

    return formatted_json


def round_values(skeleton: list) -> list:
    """round all the values of a list. Useful to save a lot of space
    when saving the skeletons to a file"""
    return [round(kpt) for kpt in skeleton]


def round_skeleton_kpts(skeleton: list) -> list:
    """round all the values of a list except every nth index. Useful
    to save a lot of space when saving the skeletons to a file"""
    for i, kpt in enumerate(skeleton):
        # if it still have confidence (beheaded or not)
        if len(skeleton) in [39, 51]:
            if i % 3 != 2:
                skeleton[i] = round(kpt)
            else:
                skeleton[i] = round(kpt, 2)

        # if it doesn't have confidence (beheaded or not)
        elif len(skeleton) in [26, 34]:
            skeleton[i] = round(kpt)

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
        return keypoints[::2], keypoints[1::2]

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


def king_of_france(keypoints: list) -> list:
    """replace the head of a skeleton by its neck"""
    period = None
    if len(keypoints) == 51:
        period = 3
    elif len(keypoints) == 34:
        period = 2

    if period is not None:
        LEar_index = 3
        REar_index = 4
        kp_LEar = keypoints[LEar_index*period:LEar_index*(period+1)]
        kp_REar = keypoints[REar_index*period:REar_index*(period+1)]
        neck_kp = create_middle_keypoint(kp_LEar, kp_REar)
        return neck_kp + keypoints[5*period:]

    raise ValueError("The skeleton is already beheaded")


def create_middle_keypoint(kp_1: list, kp_2: list):
    """create a middle keypoint from two keypoint"""
    new_kp = [0] * len(kp_1)

    for i, _ in enumerate(kp_1):
        new_kp[i] = (kp_1[i] + kp_2[i]) / 2

    return new_kp


def remove_confidence_points(keypoints: list) -> list:
    """remove confidence for every keypoints"""
    if len(keypoints) % 3 == 0:
        del keypoints[2::3]
        return keypoints

    raise ValueError("keypoints probably do not have confidence anymore",
                     "as the length is not dividable by 3")


class BK(Enum):
    """represent a skeleton body keypoints"""
    Neck = 0
    LShoulder = 1
    RShoulder = 2
    LElbow = 3
    RElbow = 4
    LWrist = 5
    RWrist = 6
    LHip = 7
    RHip = 8
    LKnee = 9
    RKnee = 10
    LAnkle = 11
    RAnkle = 12

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
                 (LShoulder, Neck),
                 (RShoulder, Neck),
                 ]
