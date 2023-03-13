"""Python API to Alphapose's bash API.

Usage:
from pathlib import Path
from tactus_data import *

input_dir = Path("C:\\Users\\marco\\Documents\\Cours\\Group Design Project - GPD\\TACTUS-data\\data\\interim\\ut_interaction\\0_1_4\\")
output_dir = Path("C:\\Users\\marco\\Documents\\Cours\\Group Design Project - GPD\\TACTUS-data\\data\\processed\\test.json")

alphapose_skeletonisation(input_dir, output_dir)
"""
import os
import sys
import json
import shutil
from pathlib import Path
from typing import Literal, Tuple
from subprocess import Popen
from contextlib import contextmanager
from PIL import Image

import alphapose


def model_weights_path(alphapose_path: Path):
    """check if the weights are downloaded. Raise an error if not"""
    weights_path = alphapose_path / "pretrained_models" / "fast_res50_256x192.pth"
    download_url = "https://drive.google.com/u/0/uc?id=1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn&export=download"

    if not weights_path.exists():
        raise FileNotFoundError(("Model weights not found. "
                                 f"You can download them at {download_url}"
                                 f"and place them in '{str(weights_path)}'."))

    return weights_path


def check_detector_weights_path(alphapose_path: Path,
                                detector: Literal["yolo", "yolox", "tracker"]
                                ):
    """check if the weights are downloaded. Raise an error if not"""
    weights_dir = alphapose_path / "detector" / detector / "data"

    if detector == "yolo":
        filename = "yolov3-spp.weights"
        download_url = "https://drive.google.com/u/0/uc?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC&export=download"
    elif detector == "tracker":
        filename = "JDE-1088x608-uncertainty"
        download_url = "https://drive.google.com/u/0/uc?id=1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA&export=download"
    elif detector == "yolox":
        filename = "yolox_x.pth"
        download_url = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.0/yolox_x.pth"

    weights_path = weights_dir / filename
    if not (weights_path).exists():
        raise FileNotFoundError(("Detector weights not found. "
                                 f"You can download them at {download_url}"
                                 f"and place them in '{str(weights_path)}'."))


def alphapose_skeletonisation(
        input_dir: Path,
        output_filepath: Path,
        detector: Literal["yolo", "yolox", "tracker"] = "yolo"
    ):
    """
    Call the alphapose bash API. AlphaPose will create a
    'alphapose-results.json' file which is renamed accordingly to the
    output_filepath value.

    Parameters
    ----------
    input_dir : Path
        Input directory with all the images to extract skeletons from
    output_filepath : Path
        Path for the result JSON file outputed by alphapose
    detector : Literal["yolo", "yolox", "tracker"], optional
        alphapose detector to use, by default "yolo"
    """
    alphapose_path = Path(alphapose.__file__).parent.parent

    config_path = Path(alphapose_path, "configs", "coco", "resnet",
                       "256x192_res50_lr1e-3_1x.yaml")
    checkpoint_path = model_weights_path(alphapose_path)

    file_path = Path(alphapose_path, "scripts", "demo_inference.py")

    arguments = [quote(file_path),
                 "--cfg", quote(config_path),
                 "--checkpoint", quote(checkpoint_path),
                 "--detector", detector,
                 "--indir", quote(input_dir),
                 "--outdir", quote(output_filepath.with_suffix('')),
                 "--pose_track"]

    with change_working_dir(alphapose_path):
        Popen(f"{sys.executable} {' '.join(arguments)}").wait()

    result_file = output_filepath.with_suffix('') / "alphapose-results.json"
    resolution = directory_resolution(input_dir)
    json_formatter(result_file, output_filepath, resolution)

    shutil.rmtree(output_filepath.with_suffix(''))


def json_formatter(input_json: Path, output_json: Path, resolution: Tuple[int, int]):
    """
    convert the json output of alphapose to a standard format for this project.
    The standard format follows the example in `data/processed/readme.md`.

    Parameters
    ----------
    input_json : Path
        the path to the output json of alphapose
    output_json : Path
        the path to where to save the new json
    resolution : List[int, int]
        the resolution of the video
    """
    alphapose_json = json.load(input_json.open())

    processed_frames = []
    standard_json = {"resolution": resolution,
                     "frames": [],}

    for skeleton in alphapose_json:
        frame_id = skeleton["image_id"]

        if not frame_id in processed_frames:
            processed_frames.append(frame_id)

            new_frame = {"frame_id": frame_id, "skeletons": []}
            standard_json["frames"].append(new_frame)

        new_skeleton = {"keypoints": skeleton["keypoints"],
                        "score": skeleton["score"],
                        "box": skeleton["box"],
                        "id": skeleton["idx"],}
        standard_json["frames"][-1]["skeletons"].append(new_skeleton)

    json.dump(standard_json, output_json.open(mode='w'))


def directory_resolution(directory: Path):
    """
    Extract the resolution of video from the directory path of all the
    extracted images of the video.

    Parameters
    ----------
    directory : Path
        The directory of all the extracted images.

    Returns
    -------
    Tuple[int, int]
        the resolution
    """
    print(directory)
    for image_path in directory.glob("*.jpg"):
        img = Image.open(image_path)
        resolution = img.size
        break
    return resolution

def quote(path: Path) -> str:
    """encapsulate a path inside double quotes"""
    if not isinstance(path, Path):
        raise TypeError("Expecting Path object")

    return '"'+str(path)+'"'


@contextmanager
def change_working_dir(path: Path):
    """temporarily change the working directory to the specified path.

    For an unknown reason, the working directory must be the one
    of alphapose."""
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)
