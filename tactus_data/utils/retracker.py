from typing import List, Tuple
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.track import Track

from tactus_data.utils.skeleton import Skeleton


def deepsort_reid(
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
    others = []
    for skeleton in skeletons:
        bbox_ltwh = cap_bbox(frame, skeleton.bbox_ltwh)
        bbs.append((bbox_ltwh, skeleton.score, "1", None))
        others.append(skeleton)

    tracks: List[Track]
    tracks = tracker.update_tracks(bbs, frame=frame, others=others)

    return tracks


def cap_bbox(frame: np.ndarray, bbox_ltrb: Tuple[float, float, float, float]):
    img_height, img_width = frame.shape[:2]
    left, top, right, bottom = bbox_ltrb
    top = min(max(top, 0), img_height - 1)
    bottom = min(max(bottom, 0), img_height - 1)
    left = min(max(left, 0), img_width - 1)
    right = min(max(right, 0), img_width - 1)

    return left, top, right, bottom


def stupid_reid(skeletons: List[Skeleton]) -> List[Skeleton]:
    """compare the x center of the bounding box to identify the
    skeleton which is the most on the left, and the one which is the
    most on the right."""
    if len(skeletons) == 0:
        return []

    x_pos_skeletons = [skeleton.bbox_cxcywh[0] for skeleton in skeletons]

    index_min = min(range(len(x_pos_skeletons)), key=x_pos_skeletons.__getitem__)
    index_max = max(range(len(x_pos_skeletons)), key=x_pos_skeletons.__getitem__)

    if len(skeletons) == 1:
        skeletons[0].tracking_id = 1
    else:
        skeletons[index_min].tracking_id = 1
        skeletons[index_max].tracking_id = 2

    return skeletons
