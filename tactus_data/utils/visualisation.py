"""
Tools to visualise skeletons in beautiful ways.

Usage
-----
from a yolov7.json

```python
from tactus_data import visualisation
from tactus_data.utils.skeletonization import keypoints_to_xy

with open(path_json) as file:
    data = json.load(file)

plot_bck_image = True

for frame in data["frames"]:
    fig, ax = plt.subplots()

    for skeleton in frame["skeletons"]:
        keypoints = keypoints_to_xy(skeleton["keypoints"])
        plot_skeleton_2d(ax, keypoints)
        if "id_stupid" in skeleton:
            plot_bbx(ax, keypoints)

    frame_path = base_frame_path / frame["frame_id"]

    if plot_bck_image:
        background_image(ax, np.asarray(Image.open(frame_path)))
    else:
        set_limits(ax, np.asarray(Image.open(frame_path)).shape[:2])

    plt.axis('off')
    plt.show()
```

Live exportation to numpy array
```python
from tactus_data import visualisation
video_resolution = (390, 270)

while True:
    skeleton_list = extract_skeleton()

    fig, ax = plt.subplots()
    for skeleton in skeleton_list:
        keypoints = keypoints_to_xy(skeleton["keypoints"])
        plot_skeleton_2d(ax, keypoints)

    set_limits(ax, video_resolution)
    np_array = fig_to_numpy(fig)
```
"""

from typing import Union, Tuple
import io

from matplotlib import pyplot as plt
from matplotlib import axes, figure
from matplotlib import patches
import numpy as np

from tactus_data.utils.skeletonization import keypoints_to_xy, skeleton_bbx, BK, \
     remove_confidence_points, king_of_france


def plot_bbx(ax: axes.Axes,
             keypoints: Union[list, tuple],
             *,
             color: str = "red",
             label: str = None):
    """
    compute and plot the bounding box of a skeleton.

    Parameters
    ----------
    ax : axes.Axes
        matplotlib ax of the current plot
    keypoints : Union[list, tuple]
        keypoints of a skeleton. Can either be a long list of
        bodykeypoints like [x1, y1, c1, x2, y2, c2, ...] or a tuple of
        x, y coordinates ([x1, x2, ...], [y1, y2, ...])
    color : str, optional
        _description_, by default "red"
    label : str, optional
        _description_, by default None
    """
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
    """
    display individual keypoints of a skeleton.

    Parameters
    ----------
    ax : axes.Axes
        matplotlib ax of the current plot
    keypoints : Union[list, tuple]
        keypoints of a skeleton. Can either be a long list of
        bodykeypoints like [x1, y1, c1, x2, y2, c2, ...] or a tuple of
        x, y coordinates ([x1, x2, ...], [y1, y2, ...])
    """
    keypoints_x, keypoints_y = keypoints_to_xy(keypoints)
    ax.scatter(keypoints_x, keypoints_y)


def plot_skeleton_2d(ax: axes.Axes,
                     keypoints: Union[list, tuple],
                     joints: list = BK.list_link.value):
    """
    draw the skeleton using lines between keypoints.

    Parameters
    ----------
    ax : axes.Axes
        matplotlib ax of the current plot
    keypoints : Union[list, tuple]
        keypoints of a skeleton. Can either be a long list of
        bodykeypoints like [x1, y1, c1, x2, y2, c2, ...] or a tuple of
        x, y coordinates ([x1, x2, ...], [y1, y2, ...])
    """
    keypoints_x, keypoints_y = keypoints_to_xy(keypoints)

    for joint_kp_1, joint_kp_2 in joints:
        ax.plot(
            [keypoints_x[joint_kp_1], keypoints_x[joint_kp_2]],
            [keypoints_y[joint_kp_1], keypoints_y[joint_kp_2]]
        )


def background_image(ax: axes.Axes, img: np.ndarray):
    """
    add an image in the plot background. You do not need to use
    set_limits afterwards.

    Parameters
    ----------
    ax : axes.Axes
        matplotlib ax of the current plot
    img : np.ndarray
        rgb image to show in the background
    """
    ax.imshow(img)


def set_limits(ax: axes.Axes, resolution: Tuple[int, int]):
    """
    You do not need to use set_limits if you added background_image.
    define x axis and y axis limits to avoid auto resize of the canva.

    Parameters
    ----------
    ax : axes.Axes
        _description_
    resolution : tuple[int, int]
        width and height of the frame
    """
    ax.set_xlim(left=0, right=resolution[1])
    ax.set_ylim(bottom=0, top=resolution[0])
    ax.set_ylim(ax.get_ylim()[::-1])


def fig_to_numpy(fig: figure.Figure) -> np.ndarray:
    """convert a matplotlib plot to a numpy array for live visualisation
    from https://stackoverflow.com/a/67823421/12550791"""
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    img = data.reshape((int(height), int(width), -1))

    plt.close(fig)
    return img

def pipeline_visualisation(ax: axes.Axes, keypoints: list):
    formatted_skel = remove_confidence_points(king_of_france(keypoints))
    plot_bbx(ax, formatted_skel,color='blue')
    scatter_skeleton_2d(ax, formatted_skel)
    plot_skeleton_2d(ax, formatted_skel)


def plot_predict(ax: axes.Axes, prediction: int,keypoints: list):
    AVAILABLE_CLASSES = ['kicking', 'punching', 'pushing', 'neutral']
    formatted_skel = remove_confidence_points(king_of_france(keypoints))
    plot_bbx(ax, formatted_skel, color='blue')
    bbx = skeleton_bbx(formatted_skel)
    ax.text(bbx[0], bbx[1], AVAILABLE_CLASSES[prediction], ha='left', va='top', color='blue')