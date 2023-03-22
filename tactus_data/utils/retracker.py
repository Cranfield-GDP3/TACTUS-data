from pathlib import Path
from typing import Union
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image


def deepsort(images_dir: Path, skeletons_json: dict):
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

    frame_to_delete = []
    for i_frame, frame in enumerate(skeletons_json["frames"]):
        frame_img = load_image(images_dir / frame["frame_id"])

        track_ids = deepsort_track_frame(tracker, frame_img, frame["skeletons"])
        if track_ids:
            for i_skeleton, skeleton in enumerate(frame["skeletons"]):
                skeleton["id_deepsort"] = track_ids[i_skeleton]
        else:
            frame_to_delete.append(i_frame)

    for i_frame in reversed(frame_to_delete):
        del skeletons_json["frames"][i_frame]

    return skeletons_json


def deepsort_track_frame(
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


def stupid_reid(skeletons_json: dict) -> dict:
    for frame in skeletons_json["frames"]:
        skeletons = frame["skeletons"]

        if len(skeletons) > 2:
            raise IndexError("There are more than 2 skeletons")

        xmin_bounding_boxes = [skeleton["box"][0]
                               for skeleton
                               in skeletons]

        if len(skeletons)==1:
            skeletons[0]["id_stupid"] = 1
        elif xmin_bounding_boxes[0] - xmin_bounding_boxes[1] > 0:
            skeletons[0]["id_stupid"] = 2
            skeletons[1]["id_stupid"] = 1
        else:
            skeletons[0]["id_stupid"] = 1
            skeletons[1]["id_stupid"] = 2

    return skeletons_json


def load_image(image_path: Path) -> np.ndarray:
    return np.asarray(Image.open(image_path))
