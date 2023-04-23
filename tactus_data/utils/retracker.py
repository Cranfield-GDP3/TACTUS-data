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


def stupid_reid(skeletons: List[Skeleton]) -> List[Skeleton]:
    """compare the x center of the bounding box to identify the
    skeleton which is the most on the left, and the one which is the
    most on the right."""
    x_pos_skeletons = [skeleton.bbox_cxcywh[0] for skeleton in skeletons]

    index_min = min(range(len(x_pos_skeletons)), key=x_pos_skeletons.__getitem__)
    index_max = max(range(len(x_pos_skeletons)), key=x_pos_skeletons.__getitem__)

    if len(skeletons) == 1:
        skeletons[0].tracking_id = 1
    else:
        skeletons[index_min].tracking_id = 1
        skeletons[index_max].tracking_id = 2

    return skeletons


def load_image(image_path: Path) -> np.ndarray:
    return np.asarray(Image.open(image_path))
