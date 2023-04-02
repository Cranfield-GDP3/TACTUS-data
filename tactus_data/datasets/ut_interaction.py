"""hanldes operations relative to the UT interaction dataset.
https://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html"""

import zipfile
import io
import requests

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
