import json
from pathlib import Path
from typing import Union
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image


def retrack(images_dir: Path, skeleton_file: Path):
    with skeleton_file.open(encoding="utf-8", mode="r") as json_file_stream:
        pose_estimation_json = json.load(json_file_stream)

    tracker = DeepSort(max_age=5)

    for frame in pose_estimation_json["frames"]:
        frame_img = load_image(images_dir / frame["frame_id"])

        if track_ids:=track_frame(tracker, frame_img, frame["skeletons"]):
            for i, _ in enumerate(frame["skeletons"]):
                frame["skeletons"][i]["id_deepsort"] = track_ids[i]

    with skeleton_file.with_stem("alphapose_2d_tracked").open(encoding="utf-8", mode="w") as fp:
        json.dump(pose_estimation_json, fp)


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
        track_ids.append(track.track_id)

    if not at_least_one_tracked:
        return False

    return track_ids


def load_image(image_path: Path) -> np.ndarray:
    return np.asarray(Image.open(image_path))
