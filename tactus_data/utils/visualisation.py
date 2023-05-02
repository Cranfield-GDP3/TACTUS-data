"""
Tools to visualise skeletons in beautiful ways.
"""

from typing import Tuple, Union
import cv2
import numpy as np
from .skeleton import Skeleton


def plot_bbox(image: np.ndarray,
              skeleton: Skeleton,
              *,
              color: Union[str, Tuple[int, int, int]] = "red",
              thickness: float = 2,
              label: str = None):
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
    """
    x_left, y_bottom, x_right, y_top = skeleton.bbox_lbrt

    cv2.rectangle(image, (x_left, y_bottom), (x_right, y_top), color=color, thickness=thickness)

    if label is not None:
        cv2.putText(image, label, (x_left, y_top),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=8, color=color)
