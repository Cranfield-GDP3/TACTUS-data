from pathlib import Path
from typing import Union
import logging
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image


def retrack(images_dir: Path, skeletons_json: dict):
    """
    retrack an extracted video with deepsort algorithm.

    Parameters
    ----------
    images_dir : Path
        The path to the folder containing all the extracted images
    skeletons_json : dict
        formatted dictionnary with the skeletons information
    """
    tracker = DeepSort(n_init=3, max_age=5)

    for frame in skeletons_json["frames"]:
        frame_img = load_image(images_dir / frame["frame_id"])

        if track_ids:=track_frame(tracker, frame_img, frame["skeletons"]):
            for i, _ in enumerate(frame["skeletons"]):
                try:
                    frame["skeletons"][i]["id_deepsort"] = track_ids[i]
                except IndexError as exc:
                    logging.warning("Deepsort retracked less skeletons than the pose detection model...")
                    raise IndexError from exc

    return skeletons_json


def track_frame(
        tracker: DeepSort,
        frame: np.ndarray,
        skeletons: list[dict]
    ) -> Union[np.ndarray, bool]:
    """
    update the tracker for one frame.

    Parameters
    ----------
    tracker : DeepSort
        the tracker object
    frame : np.ndarray
        the numpy array of the image which the skeletons have been
        extracted from
    skeletons : list[dict]
        the list of the skeletons for this frame

    Returns
    -------
    bool | np.ndarray
        returns false if no skeletons are being tracked on the frame
        else return the new track ids
    """
    bbs = []
    for skeleton in skeletons:
        bbs.append((skeleton["box"], 1, "1", None))

    tracks = tracker.update_tracks(bbs, frame=frame)

    at_least_one_tracked = False
    track_ids = []
    for track in tracks:
        if not track.is_confirmed():
            continue

        at_least_one_tracked = True
        track_ids.append(int(track.track_id))

    if not at_least_one_tracked:
        return False

    return track_ids


def load_image(image_path: Path) -> np.ndarray:
    return np.asarray(Image.open(image_path))
