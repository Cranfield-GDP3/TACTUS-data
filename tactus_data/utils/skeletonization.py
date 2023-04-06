from pathlib import Path
from enum import Enum
from typing import Union
from collections import deque
import numpy as np
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
            keypoints = skeleton["keypoints"]
            keypoints = king_of_france(keypoints)
            keypoints = round_keypoints(keypoints)
            keypoints = remove_confidence_points(keypoints)
            skeletons[i]["keypoints"] = keypoints

        frame_json["skeletons"] = skeletons
        formatted_json["frames"].append(frame_json)

    formatted_json["min_nbr_skeletons"] = min_nbr_skeletons
    formatted_json["max_nbr_skeletons"] = max_nbr_skeletons
    formatted_json["resolution"] = img.shape[:2]

    return formatted_json


def round_list(list_to_round: list) -> list:
    """round all the values of a list. Useful to save a lot of space
    when saving the skeletons to a file"""
    return [round(value) for value in list_to_round]


def round_keypoints(keypoints: list) -> list:
    """round all the values of a list except every nth index. Useful
    to save a lot of space when saving the skeletons to a file"""
    for i, kpt in enumerate(keypoints):
        # if it still have confidence (beheaded or not)
        if len(keypoints) in [39, 51]:
            if i % 3 != 2:
                keypoints[i] = round(kpt)
            else:
                keypoints[i] = round(kpt, 2)

        # if it doesn't have confidence (beheaded or not)
        elif len(keypoints) in [26, 34]:
            keypoints[i] = round(kpt)

    return keypoints


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

    return min_x, min_y, (max_x - min_x), (max_y - min_y)


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
        kp_LEar = keypoints[LEar_index * period:(LEar_index + 1) * period]
        kp_REar = keypoints[REar_index * period:(REar_index + 1) * period]
        neck_kp = middle_keypoint(kp_LEar, kp_REar)
        return neck_kp + keypoints[5 * period:]

    raise ValueError("The skeleton is already beheaded")


def middle_keypoint(kp_1: Union[list, np.ndarray], kp_2: Union[list, np.ndarray],):
    """create a middle keypoint from two keypoint"""
    if isinstance(kp_1, np.ndarray) and isinstance(kp_1, np.ndarray):
        new_kp = np.mean([kp_1, kp_2], axis=0)
    else:
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

    LKnee_angle = (LHip, LKnee, LAnkle)
    RKnee_angle = (RHip, RKnee, RAnkle)
    LElbow_angle = (LShoulder, LElbow, LWrist)
    RElbow_angle = (RShoulder, RElbow, RWrist)
    LShoulder_angle = (RShoulder, LShoulder, LElbow)
    RShoulder_angle = (LShoulder, RShoulder, RElbow)
    LHip_angle = (RHip, LHip, LKnee)
    RHip_angle = (LHip, RHip, RKnee)

    BASIC_ANGLE_LIST = [LKnee_angle, RKnee_angle,
                        LElbow_angle, RElbow_angle]
    MEDIUM_ANGLE_LIST = [LKnee_angle, RKnee_angle,
                         LElbow_angle, RElbow_angle,
                         LShoulder_angle, RShoulder_angle,
                         LHip_angle, RHip_angle]


def get_joint(keypoints: np.ndarray, kp_name: Union[BK, int]):
    """get x,y coordinates of from a keypoint name"""
    if isinstance(kp_name, BK):
        kp_name = kp_name.value

    return keypoints[2 * kp_name: 2 * kp_name + 2]


def skeleton_height(keypoints: np.ndarray):
    """return a skeleton height using the distance from the neck
    to the ankles"""
    kp_neck = get_joint(keypoints, BK.Neck)

    kp_l_ankle = get_joint(keypoints, BK.LAnkle)
    kp_r_ankle = get_joint(keypoints, BK.RAnkle)
    kp_mid_ankle = middle_keypoint(kp_l_ankle, kp_r_ankle)

    height = np.linalg.norm(kp_neck - kp_mid_ankle)
    return height


def three_points_angle(p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> float:
    """
    compute an angle from 3 points.

    Parameters
    ----------
    p1 : _type_
        _description_
    p2 : _type_
        _description_
    p3 : _type_
        _description_

    Returns
    -------
    float
        angle between (p1, p2) and (p2, p3)
    """
    if np.allclose(p1, p2) or np.allclose(p2, p3) or np.allclose(p1, p3):
        return 0

    ba = p1 - p2
    bc = p3 - p2

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def compute_angles(keypoints: list, angle_list: list[tuple[int, int, int]]) -> list[float]:
    """
    compute angles between 3 keypoints.

    Parameters
    ----------
    angle_list : list[tuple[int, int, int]]
        List of three-keypoint-indexes to compute angle for. You can use
        preexisting lists from the BK class. e.g. BK.BASIC_ANGLE_LIST or
        BK.MEDIUM_ANGLE_LIST. You can also use joints from the BK class
        e.g. [LKnee_angle, LElbow_angle, LShoulder_angle, LHip_angle].

    Example
    -------
    compute_angles(keypoints, [(BK.LHip, BK.LKnee, BK.LAnkle)])
    # will compute the 2D angle between (BK.LHip, BK.LKnee) and
    (BK.LHip, BK.LKnee).
    """
    angles = [0] * len(angle_list)

    for i, (name_kp_1, name_kp_2, name_kp_3) in enumerate(angle_list):
        kp_1 = get_joint(keypoints, name_kp_1)
        kp_2 = get_joint(keypoints, name_kp_2)
        kp_3 = get_joint(keypoints, name_kp_3)

        angles[i] = three_points_angle(kp_1, kp_2, kp_3)

    return angles


def offset_keypoints(keypoints: np.ndarray):
    x_offset, y_offset = get_joint(keypoints, BK.Neck)
    keypoints[0::2] = keypoints[0::2] - x_offset
    keypoints[1::2] = keypoints[1::2] - y_offset

    return keypoints


class SkeletonRollingWindow:
    def __init__(self, window_size: int, angles_to_compute: list = None):
        self.window_size = window_size

        if angles_to_compute is None:
            angles_to_compute = BK.BASIC_ANGLE_LIST.value
        if isinstance(angles_to_compute, BK):
            angles_to_compute = angles_to_compute.value
        self.angles_to_compute = angles_to_compute

        self.keypoints_rw = deque(maxlen=window_size)
        self.height_rw = deque(maxlen=window_size)
        self.angles_rw = deque(maxlen=window_size)
        self.velocities_rw = deque(maxlen=window_size)

    def add_skeleton(self, skeleton: dict):
        """
        add and process a new skeleton to the rolling window.

        Parameters
        ----------
        skeleton : dict
            a skeleton dictionnary containing at least "keypoints"

        Returns
        -------
        (normalized_keypoints, angles, velocities) : tuple[np.ndarray,
        np.ndarray, np.ndarray]
            return all the new information computed with the new
            skeleton.
        """
        keypoints = king_of_france(skeleton["keypoints"])
        keypoints = round_keypoints(keypoints)
        keypoints = remove_confidence_points(keypoints)
        keypoints = np.array(keypoints)

        normalized_keypoints = self._add_keypoints(keypoints)
        angles = self._add_angles()
        velocities = self._add_velocity()

        return normalized_keypoints, angles, velocities

    def add_cur_skeleton(self, keypoints: list):
        """for API compatibility"""
        self.add_skeleton(keypoints)

        return self.get_features()

    def _add_keypoints(self, keypoints: list) -> list[float]:
        """add relative keypoints to the rolling window"""
        relative_keypoints = offset_keypoints(keypoints)

        self._add_height(keypoints)
        mean_height = self.get_mean_height()

        normalized_keypoints = relative_keypoints / mean_height
        self.keypoints_rw.append(normalized_keypoints)

        return normalized_keypoints

    def _add_height(self, keypoints: list) -> float:
        """add the height over the rolling window"""
        height = skeleton_height(keypoints)
        self.height_rw.append(height)

        return height

    def _add_angles(self) -> list[float]:
        """add specified angles to the rolling window. See add_skeleton()
        for information about angles_to_compute"""
        angles = compute_angles(self.keypoints_rw[-1], self.angles_to_compute)

        self.angles_rw.append(angles)

        return angles

    def _add_velocity(self) -> list[float]:
        """add keypoints velocity to the rolling window"""
        velocity = np.zeros(len(self.keypoints_rw[0]))

        if len(self.velocities_rw) > 1:
            velocity = self.keypoints_rw[-1] - self.keypoints_rw[-2]

        self.velocities_rw.append(velocity)

        return velocity

    def get_features(self) -> tuple[bool, np.ndarray]:
        """
        return the keypoints, angles and velocities for a skeleton
        only if the window is full

        Returns
        -------
        (success, features) : bool, np.ndarray
            return the success and the features as an numpy array. if
            the window is not full, returns (False, None)
        """
        if len(self.keypoints_rw) == self.window_size:
            poses = self.get_poses_flatten()
            angles = self.get_angles_flatten()
            velocities = self.get_velocities_flatten()
            print(poses)
            print(angles)
            print(velocities)

            features = np.concatenate((poses, angles, velocities))
            return True, features

        return False, None

    def get_poses_flatten(self):
        return np.array(self.keypoints_rw).flatten()

    def get_angles_flatten(self):
        return np.array(self.angles_rw).flatten()

    def get_velocities_flatten(self):
        return np.array(self.velocities_rw).flatten()

    def get_mean_height(self):
        return np.mean(self.height_rw)
