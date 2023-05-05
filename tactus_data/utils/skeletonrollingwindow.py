from typing import List, Tuple, Sequence
from collections import deque
import numpy as np
from .skeleton import BodyAngles, Skeleton


class SkeletonRollingWindow:
    def __init__(self, window_size: int, angles_to_compute: Sequence[BodyAngles] = None):
        self.window_size = window_size

        if angles_to_compute is None:
            angles_to_compute = []
        self.angles_to_compute = angles_to_compute

        self.skeleton = None
        self.keypoints_rw = deque(maxlen=window_size)
        self.height_rw = deque(maxlen=window_size)
        self.angles_rw = deque(maxlen=window_size)
        self.velocities_rw = deque(maxlen=window_size)
        self.is_duplicated_rw = deque(maxlen=window_size)

    def add_skeleton(self, skeleton: Skeleton) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        add and process a new skeleton to the rolling window.

        Parameters
        ----------
        skeleton : Skeleton
            a skeleton object containing at least "keypoints"

        Returns
        -------
        (normalized_keypoints, angles, velocities) : Tuple[np.ndarray,
        np.ndarray, np.ndarray]
            return all the new information computed with the new
            skeleton.
        """
        self.skeleton = skeleton

        normalized_keypoints = self._add_keypoints(skeleton)
        angles = self._add_angles()
        velocities = self._add_velocity()
        self.is_duplicated_rw.append(False)

        return normalized_keypoints, angles, velocities

    def duplicate_last_entry(self, new_bbox_lbrt: Tuple[float, float, float, float]):
        """
        duplicate the last entry of the rolling window.
        """
        self.skeleton = Skeleton(bbox_lbrt=new_bbox_lbrt)
        self.keypoints_rw.append(self.keypoints_rw[-1])
        self.height_rw.append(self.height_rw[-1])
        self.angles_rw.append(self.angles_rw[-1])
        self.velocities_rw.append(self.velocities_rw[-1])
        self.is_duplicated_rw.append(True)

    def is_duplicated(self) -> bool:
        """return wether or not the last entry is a duplicated entry."""
        return self.is_duplicated_rw[-1]

    def _add_keypoints(self, skeleton: Skeleton) -> List[float]:
        """add relative keypoints to the rolling window"""
        self._add_height(skeleton.height)

        relative_keypoints = skeleton.relative_to_neck().flatten()
        mean_height = self._get_mean_height()
        for i in range(len(relative_keypoints)):
            relative_keypoints[i] = relative_keypoints[i] / mean_height

        self.keypoints_rw.append(relative_keypoints)

        return relative_keypoints

    def _add_height(self, height: int) -> int:
        """add the height over the rolling window"""
        self.height_rw.append(height)

    def _get_mean_height(self) -> float:
        """return the average height of the skeleton on the window."""
        return sum(self.height_rw) / len(self.height_rw)

    def _add_angles(self) -> List[float]:
        """add specified angles to the rolling window. See add_skeleton()
        for information about angles_to_compute"""
        angles = self.skeleton.get_angles(self.angles_to_compute)

        self.angles_rw.append(angles)

        return angles

    def _add_velocity(self) -> List[float]:
        """add keypoints velocity to the rolling window"""
        velocity = [0] * len(self.keypoints_rw[0])

        if len(self.keypoints_rw) > 1:
            for i in range(len(velocity)):
                velocity[i] = self.keypoints_rw[-1][i] - self.keypoints_rw[-2][i]

        self.velocities_rw.append(velocity)

        return velocity

    def get_features(self) -> Tuple[bool, np.ndarray]:
        """
        return the keypoints, angles and velocities for a skeleton
        only if the window is full

        Returns
        -------
        (success, features) : bool, np.ndarray
            return the success and the features as an numpy array. if
            the window is not full, returns (False, None)
        """
        if len(self.keypoints_rw) == self.window_size:
            features = self.keypoints_rw + self.angles_rw + self.velocities_rw

            features = np.array(features).flatten()
            return True, features

        return False, None
