from pathlib import Path
from enum import Enum
import random
import json
from tqdm import tqdm
from tactus_yolov7 import Yolov7

from tactus_data.utils import video_to_img
from tactus_data.utils.retracker import stupid_reid
from tactus_data.utils.skeletonization import yolov7
from tactus_data.utils.skeletonization import MODEL_WEIGHTS_PATH
from tactus_data.utils.data_augment import grid_augment

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

    progress_bar = tqdm(iterable=input_dir.rglob(f"*.{video_extension}"), total=nbr_of_videos)
    for video_path in progress_bar:
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

    model = Yolov7(MODEL_WEIGHTS_PATH, device)

    progress_bar = tqdm(iterable=input_dir.glob(f"*/{fps_folder_name}"), total=nbr_of_videos)
    for extracted_frames_dir in progress_bar:
        video_name = extracted_frames_dir.parent.name
        skeletons_output_dir: Path = (output_dir / dataset.name / video_name
                                      / fps_folder_name / "yolov7.json")

        formatted_json = yolov7(extracted_frames_dir, model)
        tracked_json = stupid_reid(formatted_json)
        filtered_json = _delete_skeletons_keys(tracked_json, ["box", "score"])

        skeletons_output_dir.parent.mkdir(parents=True, exist_ok=True)
        with skeletons_output_dir.open(encoding="utf-8", mode="w") as fp:
            json.dump(filtered_json, fp)


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


def augment_all_vid(input_folder_path: Path,
                    grid: list[list[list]],
                    fps: int,
                    json_name: str = "yolov7.json",
                    max_copy: int = -1,
                    random_seed: int = 30000):
    """
    Generate 1 json with a new scaling of skeletons. The distance
    parameter allows you to virtually move the camera further or closer
    to the frame so that the scale change accordingly, distance is in
    meters represent the distance added on the original position of the
    camera, if the camera goes closer(-) / further(+).

    Parameters
    ----------
    input_folder_path : Path,
                path of the folder where the original json are located
    grid : list[list[list]],
           the grid of value that will be use for the grid search the
           gris is a list of list, each element of the list a list of
           all the argument of each augment can take without including
           the path, in the form of another list :

           [list_flip_h, list_camera_distance_2d,list_rotation_2d,
           list_noise_2d]

           Ex:
           list_flip_h = [True, False]
           list_camera_distance_2d =[distance,focal_length]
           list_rotation_2d = [max_angle,num_copy,rotate_center]
           list_noise_2d = [num_copy, noise_magnitude]
           If you don't want an augment to be used just put an empty
           list
           list_camera_distance_2d :
            - distance = [-5,-2, 2, 5, 10, 15]
            - focal_length = [2.8, 3.6, 4.0, 6.0]
            Grid for testing : [[True, False], [[0, 5], [3.6]],
                               [[-10, 0, 10],[1], [(BK.LAnkle,
                               BK.RAnkle)]], [[1], [0, 4]]]
    fps : int,
          pick the fps folder you want to augment for each video
          (the fps folder must exist)
    json_name : str,
                name of the json file in each video folder.
    max_copy : int,
               Number of copy of the original file are going to be
               generated
    random_seed : value of the random seed to replicated same training data
    """
    random.seed(random_seed)
    total_cpy = 0
    list_dir = list(input_folder_path.iterdir())
    list_dir.remove(input_folder_path / "readme.md")
    for path_dir in tqdm(list_dir):
        vid_path = Path(path_dir / _fps_folder_name(fps))
        vid_name = vid_path.glob('**/' + json_name)
        for injson in vid_name:
            total_cpy += grid_augment(injson, grid, max_copy)
