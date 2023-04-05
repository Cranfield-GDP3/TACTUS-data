"""hanldes operations relative to the UT interaction dataset.
https://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html"""

import zipfile
import io
import requests
import random
from pathlib import Path

from tactus_data.datasets import dataset

NAME = dataset.NAMES.ut_interaction

DOWNLOAD_URL = [
    "http://cvrc.ece.utexas.edu/SDHA2010/videos/competition_1/ut-interaction_segmented_set1.zip",
    "http://cvrc.ece.utexas.edu/SDHA2010/videos/competition_1/ut-interaction_segmented_set2.zip"
]

ACTION_INDEXES = ["neutral", "neutral", "kicking",
                  "neutral", "punching", "pushing"]


def extract_frames(fps: int = 10):
    """
    Extract frame from a folder containing videos.

    Parameters
    ----------
    fps : int
        The fps we want to have
    """
    dataset.extract_frames(NAME, dataset.RAW_DIR, dataset.INTERIM_DIR, fps, "avi")


def extract_skeletons(fps: int = 10, device: str = None):
    """
    Extract skeletons from a folder containing video frames using
    yolov7.

    Parameters
    ----------
    fps : int
        the fps of the extracted frames
    device : str
        the computing device to use with yolov7.
        Can be 'cpu', 'cuda:0' etc.
    """
    dataset.extract_skeletons(NAME, dataset.INTERIM_DIR, dataset.PROCESSED_DIR, fps, device)


def augment(grid: dict = None, fps: int = 10):
    """augment all skeletons of ut_interaction"""
    dataset.augment_all_vid(dataset.PROCESSED_DIR / NAME.name, grid, fps)


def download():
    """Download and extract dataset from source"""
    for zip_file_url in DOWNLOAD_URL:
        response = requests.get(zip_file_url, timeout=1000)

        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_response:
            zip_response.extractall(dataset.RAW_DIR / NAME.name)


def label_from_video_name(video_name: str) -> str:
    """
    Extract the label name from the video name. The video name
    should have the format `{sequence}_{sample}_{label}`. It
    must no include the file extension.

    Parameters
    ----------
    video_name : str
        The name of the video to extract a label from.

    Returns
    -------
    str :
        the corresponding label
    """
    _, _, action = video_name.split("_")
    label = ACTION_INDEXES[int(action)]

    return label


def data_split(ut_interaction_dir: Path,
               split_strategy: tuple = (80, 10, 10),
               random_seed: int = 30000) -> list:
    """
    Randomly split the data between train/val/test following a
    split_strategy defined in the parameters

    Parameters
    ----------
    ut_interaction_dir : Path,
        Path of the processed directory containing all the video folder
    split_strategy : tuple,
        % of value for each part of the data between train validation
        and test, the sum of the tuple must be equal to 100
    random_seed : int,
        value of the random seed to replicated same data split

    Returns
    -------
    list[list[Path] :
        List composed of the list of Path for all video folder on
        train/validation/test
    """
    repartition = [[], [], [], []]
    list_dir = sorted(ut_interaction_dir.iterdir())
    for i in list_dir:
        folder_name = i.stem
        vid_label = label_from_video_name(folder_name)
        if vid_label == 'kicking':
            repartition[0].append(i)
        elif vid_label == 'punching':
            repartition[1].append(i)
        elif vid_label == 'pushing':
            repartition[2].append(i)
        else:
            repartition[3].append(i)
    class_repartition = [len(repartition[0]), len(repartition[1]), len(repartition[2]), len(repartition[3])]
    # randomize
    random.seed(random_seed)
    random.shuffle(repartition[0])
    random.shuffle(repartition[1])
    random.shuffle(repartition[2])
    random.shuffle(repartition[3])
    # split data
    split_data = [[], [], []]
    # train
    for j in range(len(class_repartition)):
        num = class_repartition[j] * split_strategy[0] // 100
        current_num = 0
        while current_num < num:
            split_data[0].append(repartition[j][current_num])
            current_num += 1
    # val
    for j in range(len(class_repartition)):
        num = class_repartition[j] * (split_strategy[0] + split_strategy[1]) // 100
        current_num = class_repartition[j] * split_strategy[0] // 100
        while current_num < num:
            split_data[1].append(repartition[j][current_num])
            current_num += 1
    # test
    for j in range(len(class_repartition)):
        num = class_repartition[j]
        current_num = class_repartition[j] * (split_strategy[0] + split_strategy[1]) // 100
        while current_num < num:
            split_data[2].append(repartition[j][current_num])
            current_num += 1
    return split_data
