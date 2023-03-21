from pathlib import Path
from enum import Enum
import json
import logging
from tqdm import tqdm
from tactus_yolov7 import Yolov7

from tactus_data.utils import video_to_img
from tactus_data.utils.retracker import deepsort
from tactus_data.utils.yolov7 import yolov7
from tactus_data.utils.yolov7 import MODEL_WEIGHTS_PATH

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
    nbr_of_videos = _count_files_in_dir(input_dir, f"*.{video_extension}")

    for video_path in tqdm(iterable=input_dir.rglob(f"*.{video_extension}"), total=nbr_of_videos):
        frame_output_dir = (output_dir
                            / dataset.name
                            / video_path.stem
                            / _fps_folder_name(fps))

        video_to_img.extract_frames(video_path, frame_output_dir, fps)

def extract_skeletons(
        dataset: NAMES,
        input_dir: Path,
        output_dir: Path,
        fps: int,
        device: str
    ):
    """
    Extract skeletons from a folder containing video frames using
    yolov7.

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
    device : str
        the computing device to use with yolov7.
        Can be 'cpu', 'cuda:0' etc.
    """
    input_dir = input_dir / dataset.name
    fps_folder_name = _fps_folder_name(fps)

    nbr_of_videos = _count_files_in_dir(input_dir, f"*/{fps_folder_name}")

    discarded_videos = []

    model = Yolov7(MODEL_WEIGHTS_PATH, device)

    for extracted_frames_dir in tqdm(iterable=input_dir.glob(f"*/{fps_folder_name}"), total=nbr_of_videos):
        video_name = extracted_frames_dir.parent.name
        skeletons_output_dir: Path = (output_dir
                                      / dataset.name
                                      / video_name
                                      / fps_folder_name
                                      / "yolov7.json")

        formatted_json = yolov7(extracted_frames_dir, model)

        try:
            tracked_json = deepsort(extracted_frames_dir, formatted_json)
        except IndexError:
            discarded_videos.append(video_name)
        else:
            filtered_json = _delete_skeletons_keys(tracked_json, ["box", "score"])

            skeletons_output_dir.parent.mkdir(parents=True, exist_ok=True)
            with skeletons_output_dir.open(encoding="utf-8", mode="w") as fp:
                json.dump(filtered_json, fp)

    if len(discarded_videos) > 0:
        logging.warning("Discarding %s videos from %s:", len(discarded_videos), dataset.name)
        logging.warning("\t %s", "\t".join(discarded_videos))


def _fps_folder_name(fps: int):
    """return the name of the fps folder for a given fps value"""
    return f"{fps}fps"


def _delete_skeletons_keys(formatted_json: dict, keys_to_remove: list[str]):
    """remove every keys specified from the skeleton dictionnary"""
    for frame in formatted_json["frames"]:
        for skeleton in frame["skeletons"]:
            for key in keys_to_remove:
                del skeleton[key]

    return formatted_json

def _count_files_in_dir(directory: Path, pattern: str):
    nbr_of_files = 0
    for _ in directory.rglob(pattern):
        nbr_of_files += 1

    return nbr_of_files
