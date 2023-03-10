from pathlib import Path
from enum import Enum

from tactus_data.utils import video_to_img
from tactus_data.utils.alphapose import alphapose_skeletonisation

RAW_DIR = Path("data/raw/")
INTERIM_DIR = Path("data/interim/")
PROCESSED_DIR = Path("data/processed/")
NAMES = Enum('NAMES', ['ut_interaction'])

def extract_frames(
        dataset: NAMES,
        input_dir: Path,
        output_dir: Path,
        fps: int,
        video_extension: str
    ):
    """
    Extract frame from a folder containing videos.

    Parameters
    ----------
    dataset : NAMES
        the Enum name of the dataset. Accessible through
        `NAMES.dataset_name`
    input_dir : Path
        The path to the raw folder containing the raw datasets
    output_dir : Path
        The interim folder containing the dataset folders which contain
        extracted frames
    fps : int
        The fps we want to have
    video_extension : str
        The video extensions (avi, mp4, etc.)
    """
    input_dir = input_dir / dataset.name

    for video_path in input_dir.rglob(f"*.{video_extension}"):
        frame_output_dir = (output_dir
                            / dataset.name
                            / video_path.stem
                            / _fps_folder_name(fps))

        video_to_img.extract_frames(video_path, frame_output_dir, fps)

def extract_skeletons(
        dataset: NAMES,
        input_dir: Path,
        output_dir: Path,
        fps: int
    ):
    """
    Extract skeletons from a folder containing video frames using
    alphapose.

    Parameters
    ----------
    dataset : NAMES
        the Enum name of the dataset. Accessible through
        `NAMES.dataset_name`
    input_dir : Path
        The interim folder containing the dataset folders which contain
        extracted frames
    output_dir : Path
        the processed folder. Will be saved under
        `output_dir/dataset_name/fps/name.json`
    fps : int
        the fps of the extracted frames
    """
    input_dir = input_dir / dataset.name
    fps_folder_name = _fps_folder_name(fps)

    for extracted_frames_dir in input_dir.glob(f"*/{fps_folder_name}"):
        video_name = extracted_frames_dir.parent.name
        skeletons_output_dir = (output_dir
                                / dataset.name
                                / video_name
                                / fps_folder_name
                                / "alphapose_2d.json")

        alphapose_skeletonisation(extracted_frames_dir.absolute(),
                                  skeletons_output_dir)

def _fps_folder_name(fps: int):
    """return the name of the fps folder for a given fps value"""
    return f"{fps}fps"
