"""
Tools to visualise skeletons in beautiful ways.
"""

from typing import Tuple, Union
import cv2
import numpy as np
from .skeleton import Skeleton, BodyJoints


body_joint_colors = ["#13EAC9", "#04D8B2",  # ankles to knees
                     "#069AF3", "#1F88F2",  # knees to hips
                     "#7FFF00",             # left hip to right hip
                     "#C1F80A", "#AAFF32",  # hips to shoulders
                     "#FFD700",             # left shoulder to right shoulder
                     "#FFA500", "#F97306",  # shoulders to elbows
                     "#FF4500", "#FE420F",  # elbows to wrists
                     "#EFE63B", "#EDC140",  # shoulders to neck
                     ]


def plot_bbox(image: np.ndarray,
              skeleton: Skeleton,
              *,
              color: Union[str, Tuple[int, int, int]] = "red",
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
    x_left, y_bottom, x_right, y_top = skeleton.bbox_lbrt

    cv2.rectangle(image, (x_left, y_bottom), (x_right, y_top), color=color, thickness=thickness)

    if label is not None:
        cv2.putText(image, label, (x_left, y_top),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=8, color=color)

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

        cv2.line(image, skeleton.get_kpt(kp_1), skeleton.get_kpt(kp_2), joint_color, thickness)

    return image
