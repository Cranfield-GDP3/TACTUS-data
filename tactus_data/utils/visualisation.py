"""
Tools to visualise skeletons in beautiful ways.
"""

from typing import Tuple, Union
import cv2
import numpy as np
from .skeleton import Skeleton, BodyJoints, round_list


body_joint_colors = [(19, 234, 201), (4, 216, 178),   # ankles to knees
                     (6, 154, 243), (31, 136, 242),   # knees to hips
                     (127, 255, 0),                   # left hip to right hip
                     (193, 248, 10), (170, 255, 50),  # hips to shoulders
                     (255, 215, 0),                   # left shoulder to right shoulder
                     (255, 165, 0), (249, 115, 6),    # shoulders to elbows
                     (255, 69, 0), (254, 66, 15),     # elbows to wrists
                     (239, 230, 59), (237, 193, 64),  # shoulders to neck
                     ]


def plot_bbox(image: np.ndarray,
              skeleton: Skeleton,
              *,
              color: Tuple[int, int, int] = (0, 0, 255),
              thickness: float = 2,
              label: str = None
              ) -> np.ndarray:
    """
    plot the bounding box of a skeleton.

    Parameters
    ----------
    image : np.ndarray
        extracted image from a camera, or a blank background.
    skeleton : Skeleton
        skeleton object. Its bounding box will be used if set. Otherwise,
        the bounding box will be estimated from the skeleton keypoints.
    color : str | Tuple[int, int, int], optional
        color of the bounding box. Can be a BGR tuple or a string,
        by default "red".
    thickness : float
        thickness of the bounding box.
    label : str, optional
        name of the label, by default None.

    Returns
    -------
    np.ndarray
        image with joints drawn.
    """
    x_left, y_bottom, x_right, y_top = round_list(skeleton.bbox_lbrt)

    image = cv2.rectangle(image,
                          (x_left, y_bottom),
                          (x_right, y_top),
                          color=color, thickness=thickness)

    if label is not None:
        image = cv2.putText(image, label, (x_left, y_top),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=color)

    return image


def plot_joints(image: np.ndarray,
                skeleton: Skeleton,
                *,
                thickness: float = 2,
                color: Union[str, Tuple[int, int, int]] = None
                ) -> np.ndarray:
    """
    draw lines that represent skeleton joints.

    Parameters
    ----------
    image : np.ndarray
        extracted image from a camera, or a blank background.
    skeleton : Skeleton
        the skeleton object with its keypoints.
    thickness : float, optional
        thickness of the joint lines, by default 2.
    color : Union[str, Tuple[int, int, int]], optional
        color of the lines. Can be a BGR tuple or a string,
        by default None which gives a different color to each joints.

    Returns
    -------
    np.ndarray
        image with joints drawn.
    """
    joint_color = color
    for i, (kp_1, kp_2) in enumerate(BodyJoints):
        if color is None:
            joint_color = body_joint_colors[i]

        image = cv2.line(image,
                         round_list(skeleton.get_kpt(kp_1)),
                         round_list(skeleton.get_kpt(kp_2)),
                         joint_color, thickness)

    return image
