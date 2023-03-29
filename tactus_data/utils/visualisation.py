from typing import Union

from matplotlib import axes
from matplotlib import patches
import numpy as np

from tactus_data.utils.skeletonization import keypoints_to_xy, skeleton_bbx


def plot_bbx(ax: axes.Axes, keypoints: Union[list, tuple], color: str = "red", label: str = None):
    """compute and plot the bounding box of a skeleton. A label can be added"""
    x_left, y_bottom, width, height = skeleton_bbx(keypoints)

    box = patches.Rectangle(
        (x_left, y_bottom), width, height,
        edgecolor=color, facecolor="none"
    )
    ax.add_patch(box)

    if label is not None:
        l = ax.annotate(label, (x_left+width, y_bottom+height),
                        fontsize=8, color="white", ha="right", va="bottom")
        l.set_bbox({"facecolor": color, "alpha": 0.8, "edgecolor": color})


def scatter_skeleton_2d(ax: axes.Axes, keypoints: Union[list, tuple]):
    """display individual keypoints of a skeleton"""
    keypoints_x, keypoints_y = keypoints_to_xy(keypoints)
    ax.scatter(keypoints_x, keypoints_y)


def plot_skeleton_2d(ax: axes.Axes, keypoints: Union[list, tuple], joints: list):
    """draw the skeleton using lines between keypoints"""
    keypoints_x, keypoints_y = keypoints_to_xy(keypoints)

    for (joint_kp_1, joint_kp_2) in joints:
        ax.plot(
            [keypoints_x[joint_kp_1], keypoints_x[joint_kp_2]],
            [keypoints_y[joint_kp_1], keypoints_y[joint_kp_2]]
        )


def background_image(ax: axes.Axes, img: np.ndarray):
    """add an image in the plot background"""
    ax.imshow(img)
