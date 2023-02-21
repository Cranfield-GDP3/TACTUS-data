"""Python API to Alphapose's bash API.

Usage:
from pathlib import Path
from tactus_data import skeleton_extraction

alphapose_path = Path("C:\\...\\AlphaPose")
input_dir = Path("C:\\...\\TACTUS-data\\data\\interim\\test")
output_dir = Path("C:\\...\\TACTUS-data\\data\\processed\\test.json")

skeleton_extraction(alphapose_path, input_dir, output_dir)
"""
import os
import sys
from pathlib import Path
from typing import Literal
from subprocess import Popen
from contextlib import contextmanager


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
        alphapose_path: Path,
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
    alphapose_path : Path
        The path to the root folder of the alphapose project
    input_dir : Path
        Input directory with all the images to extract skeletons from
    output_filepath : Path
        Path for the result JSON file outputed by alphapose
    detector : Literal["yolo", "yolox", "tracker"], optional
        alphapose detector to use, by default "yolo"
    """
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
    result_file.replace(output_filepath)


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
