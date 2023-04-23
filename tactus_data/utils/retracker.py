from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.track import Track

from tactus_data.utils.skeleton import Skeleton


def deepsort_track_frame(
    tracker: DeepSort,
    frame: np.ndarray,
    skeletons: List[Skeleton]
) -> List[Track]:
    """
    update the tracker for one frame.

    Parameters
    ----------
    tracker : DeepSort
        the tracker object
    frame : np.ndarray
        the numpy array of the image which the skeletons have been
        extracted from.
    skeletons : list[Skeleton]
        the list of the skeletons for this frame. Each skeleton must
        have a its bounding box.

    Returns
    -------
    List[Tracks]
        returns the list of deepsort tracks.
    """
    bbs = []
    for skeleton in skeletons:
        bbs.append((skeleton.bbox_ltwh, skeleton.score, "1", None))

    tracks: list[Track]
    tracks = tracker.update_tracks(bbs, frame=frame)

    return tracks


def stupid_reid(skeletons_json: dict) -> dict:
    for frame in skeletons_json["frames"]:
        skeletons = frame["skeletons"]

        x_pos_skeletons = [skeleton["keypoints"][0] # Nose keypoint
                           for skeleton
                           in skeletons]

        index_min = min(range(len(x_pos_skeletons)), key=x_pos_skeletons.__getitem__)
        index_max = max(range(len(x_pos_skeletons)), key=x_pos_skeletons.__getitem__)

        if len(skeletons) == 1:
            skeletons[0]["id_stupid"] = 1
        else:
            skeletons[index_min]["id_stupid"] = 1
            skeletons[index_max]["id_stupid"] = 2

    return skeletons_json


def load_image(image_path: Path) -> np.ndarray:
    return np.asarray(Image.open(image_path))
