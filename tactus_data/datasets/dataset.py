from pathlib import Path

from tactus_data.utils import video_to_img
from tactus_data.utils.alphapose import alphapose_skeletonisation

RAW_DIR = Path("data/raw/")
INTERIM_DIR = Path("data/interim/")
PROCESSED_DIR = Path("data/processed/")

def extract_frames(
        input_dir: Path,
        output_dir: Path,
        fps: int,
        video_extension: str
    ):
    """
    Extract frame from a folder containing videos.

    Parameters
    ----------
    input_dir : Path
        The path to the folder containing all the videos
    output_dir : Path
        The path to where to save the frames
    fps : int
        The fps we want to have
    video_extension : str
        The video extensions (avi, mp4, etc.)
    """
    for video_path in input_dir.rglob(f"*.{video_extension}"):
        frame_output_dir = (output_dir
                            / video_path.stem
                            / _fps_folder_name(fps))

        video_to_img.extract_frames(video_path, frame_output_dir, fps)

def extract_skeletons(
        input_dir: Path,
        output_dir: Path,
        fps: int
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
    fps_folder_name = _fps_folder_name(fps)

    for extracted_frames_dir in input_dir.glob(f"*/{fps_folder_name}"):
        final_dir_name = extracted_frames_dir.parent.name

        alphapose_skeletonisation(extracted_frames_dir.absolute(),
                                    output_dir.absolute()
                                    / final_dir_name
                                    / fps_folder_name
                                    / "alphapose_2d.json")

def _fps_folder_name(fps: int):
    """return the name of the fps folder for a given fps value"""
    return f"{fps}fps"
