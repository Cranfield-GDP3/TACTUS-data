from pathlib import Path
from enum import Enum
import random
import json
from tqdm import tqdm

from tactus_data.utils.yolov8 import Yolov8
from tactus_data.utils.thread_videocapture import VideoCapture
from tactus_data.utils.retracker import stupid_reid
from tactus_data.utils.data_augment import grid_augment, DEFAULT_GRID

RAW_DIR = Path("data/raw/")
PROCESSED_DIR = Path("data/processed/")
NAMES = Enum('NAMES', ['ut_interaction'])


def extract_skeletons(
    dataset: NAMES,
    fps: int,
    video_extension: str,
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
    fps : int
        the fps for the skeleton extraction
    video_extension : int
        the video extensions (avi, mp4, etc.)
    device : str
        the computing device to use with yolov7.
        Can be 'cpu', 'cuda:0' etc.
    """
    input_dir = RAW_DIR / dataset.name
    output_dir = PROCESSED_DIR / dataset.name
    fps_folder_name = _fps_folder_name(fps)

    model_skeleton = Yolov8(Path("data/models"), "yolov8x-pose-p6.pt", device)

    nbr_of_videos = _count_files_in_dir(input_dir, f"*.{video_extension}")
    progress_bar = tqdm(iterable=input_dir.rglob(f"*.{video_extension}"), total=nbr_of_videos)
    for video_path in progress_bar:
        output_path: Path = (output_dir / video_path.stem / fps_folder_name / "yolov8.json")

        video_dict = _extract_skeletons_video(model_skeleton, video_path, fps)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open(encoding="utf-8", mode="w") as fp:
            json.dump(video_dict, fp)


def _fps_folder_name(fps: int):
    """return the name of the fps folder for a given fps value"""
    return f"{fps}fps"


def _count_files_in_dir(directory: Path, pattern: str):
    nbr_of_files = 0
    for _ in directory.rglob(pattern):
        nbr_of_files += 1

    return nbr_of_files


def _extract_skeletons_video(model: Yolov8, video_path: Path, fps: int):
    cap = VideoCapture(video_path, target_fps=fps, buffer_size=1)

    video_dict = {"frames": []}

    max_number_of_skeleton = 0
    min_number_of_skeleton = float("inf")

    resolution = None
    first_frame = True

    while (cap_frame := cap.read()) is not None:
        frame_id, frame = cap_frame

        if first_frame:
            resolution = frame.shape[:2]
            first_frame = False

        frame_dict = {"frame_id": frame_id}

        skeletons = model.extract_skeletons(frame)
        skeletons = stupid_reid(skeletons)

        frame_dict["skeletons"] = skeletons

        max_number_of_skeleton = max(max_number_of_skeleton, len(skeletons))
        min_number_of_skeleton = min(min_number_of_skeleton, len(skeletons))

        video_dict["frames"].append(frame_dict)

    if resolution is not None:
        video_dict["resolution"] = resolution
        video_dict["max_nbr_skeletons"] = max_number_of_skeleton
        video_dict["min_nbr_skeletons"] = min_number_of_skeleton

        return video_dict


def augment_all_vid(input_folder_path: Path,
                    grid: dict = None,
                    fps: int = 10,
                    json_name: str = "yolov7.json",
                    random_seed: int = 30000):
    """
    Run grid_augment() which generate multiple json from an original
    json with different types of augments like translation, rotation,
    scaling on all 3 axis. For all the json files in the data/processed
    folder

    Parameters
    ----------
    input_folder_path : Path,
        path to the folder that contains the original jsons
    grid : dict,
        storing all needed parameters for augments. Available keys:
        "noise_amplitude", "horizontal_flip", "vertical_flip",
        "rotation_y", "rotation_z", "rotation_x", "scale_x", "scale_y".
        The values of these keys must be arrays. By default
        DEFAULT_GRID = {
            "noise_amplitude": np.linspace(1, 4, 2),
            "horizontal_flip": [True, False],
            "rotation_y": np.linspace(-20, 20, 3),
            "rotation_z": np.linspace(-20, 20, 3),
            "rotation_x": np.linspace(-20, 20, 3),
            "scale_x": np.linspace(0.8, 1.2, 3),
            "scale_y": np.linspace(0.8, 1.2, 3),
        }
    fps : int,
        pick the fps folder you want to augment for each video
        (the fps folder must exist)
    json_name : str,
        name of the json file in each video folder.
    random_seed : int,
        value of the random seed to replicated same training data
    """
    if grid is None:
        grid = DEFAULT_GRID

    random.seed(random_seed)

    patern = f"**/{_fps_folder_name(fps)}/{json_name}"
    nbr_of_videos = _count_files_in_dir(input_folder_path, patern)
    progress_bar = tqdm(iterable=input_folder_path.glob(patern), total=nbr_of_videos)

    for in_json in tqdm(progress_bar):
        grid_augment(in_json, grid)
