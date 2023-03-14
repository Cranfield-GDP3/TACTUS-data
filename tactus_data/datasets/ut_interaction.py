"""hanldes operations relative to the UT interaction dataset.
https://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html"""

from pathlib import Path
import zipfile
import io
import requests

from tactus_data.datasets import dataset

NAME = "ut_interaction"

DOWNLOAD_URL = [
    "http://cvrc.ece.utexas.edu/SDHA2010/videos/competition_1/ut-interaction_segmented_set1.zip",
    "http://cvrc.ece.utexas.edu/SDHA2010/videos/competition_1/ut-interaction_segmented_set2.zip"
]

ACTION_INDEXES = ["neutral", "neutral", "kicking",
                  "neutral", "punching", "pushing"]


def extract_frames(
        input_dir: Path = dataset.RAW_DIR,
        output_dir: Path = dataset.INTERIM_DIR,
        desired_fps: int = 10,
    ):
    """
    Extract frame from a folder containing videos.

    Parameters
    ----------
    input_dir : Path
        The path to the dataset folder containing all the videos
    output_dir : Path
        The path to where to save the frames
    fps : int
        The fps we want to have
    """
    input_dir = input_dir/ NAME
    output_dir = output_dir / NAME

    dataset.extract_frames(input_dir, output_dir, desired_fps, "avi")


def extract_skeletons(
        input_dir: Path = dataset.INTERIM_DIR,
        output_dir: Path = dataset.PROCESSED_DIR,
        fps: int = 10,
):
    """
    Extract skeletons from a folder containing video frames using
    alphapose.

    Parameters
    ----------
    input_dir : Path
        The folder containing the dataset folder which contains
        the extracted frames
    output_dir : Path
        the folder where the outputed file will be saved. Will be
        under output_dir/dataset_name/fps/name.json
    fps : int
        the fps of the extracted frames
    """
    input_dir = input_dir / NAME
    output_dir = output_dir / NAME

    dataset.extract_skeletons(input_dir, output_dir, fps)


def download(download_dir: Path = dataset.RAW_DIR):
    """
    Download and extract dataset from source.

    Parameters
    ----------
    download_dir : Path, optional
        The path where to download the data,
        by default dataset.RAW_DIR
    """
    for zip_file_url in DOWNLOAD_URL:
        response = requests.get(zip_file_url, timeout=1000)

        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_response:
            zip_response.extractall(download_dir / NAME)


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
