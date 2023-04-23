from typing import List, Tuple, Union, Sequence
from enum import IntEnum, Enum
import numpy as np


class BodyKpt(IntEnum):
    """
    represents a skeleton body keypoints.
    members of IntEnum are also ints, which is convenient when we use
    this keypoints as indices.
    """
    Neck = 0
    LShoulder = 1
    RShoulder = 2
    LElbow = 3
    RElbow = 4
    LWrist = 5
    RWrist = 6
    LHip = 7
    RHip = 8
    LKnee = 9
    RKnee = 10
    LAnkle = 11
    RAnkle = 12


BodyJoints = [
    (BodyKpt.RAnkle, BodyKpt.RKnee),
    (BodyKpt.LAnkle, BodyKpt.LKnee),
    (BodyKpt.RKnee, BodyKpt.RHip),
    (BodyKpt.LKnee, BodyKpt.LHip),
    (BodyKpt.RHip, BodyKpt.LHip),
    (BodyKpt.RHip, BodyKpt.RShoulder),
    (BodyKpt.LHip, BodyKpt.LShoulder),
    (BodyKpt.RShoulder, BodyKpt.LShoulder),
    (BodyKpt.RShoulder, BodyKpt.RElbow),
    (BodyKpt.RElbow, BodyKpt.RWrist),
    (BodyKpt.LShoulder, BodyKpt.LElbow),
    (BodyKpt.LElbow, BodyKpt.LWrist),
    (BodyKpt.LShoulder, BodyKpt.Neck),
    (BodyKpt.RShoulder, BodyKpt.Neck),
]


class BodyAngles(Enum):
    """
    represents a skeleton existing angles between joints
    """
    LKnee = (BodyKpt.LHip, BodyKpt.LKnee, BodyKpt.LAnkle)
    RKnee = (BodyKpt.RHip, BodyKpt.RKnee, BodyKpt.RAnkle)
    LElbow = (BodyKpt.LShoulder, BodyKpt.LElbow, BodyKpt.LWrist)
    RElbow = (BodyKpt.RShoulder, BodyKpt.RElbow, BodyKpt.RWrist)
    LShoulder = (BodyKpt.RShoulder, BodyKpt.LShoulder, BodyKpt.LElbow)
    RShoulder = (BodyKpt.LShoulder, BodyKpt.RShoulder, BodyKpt.RElbow)
    LHip = (BodyKpt.RHip, BodyKpt.LHip, BodyKpt.LKnee)
    RHip = (BodyKpt.LHip, BodyKpt.RHip, BodyKpt.RKnee)


SMALL_ANGLES_LIST = [
    BodyAngles.LKnee, BodyAngles.RKnee, BodyAngles.LElbow, BodyAngles.RElbow
]
MEDIUM_ANGLES_LIST = [
    BodyAngles.LKnee, BodyAngles.RKnee, BodyAngles.LElbow, BodyAngles.RElbow,
    BodyAngles.LShoulder, BodyAngles.RShoulder, BodyAngles.LHip, BodyAngles.RHip
]


class Skeleton:
    def __init__(self, bbox_bltr: Sequence = (), score: float = None, keypoints: Sequence = None, keypoints_visibility: Sequence = None) -> None:
        self._boundbing_box_bltr: Tuple[float, float, float, float] = None
        self._score: float = score
        self._keypoints: Tuple[float] = None
        self._keypoints_visibility: Tuple[bool] = None
        self._height: float = None

        self.bbox = bbox_bltr
        self.keypoints = keypoints
        self.keypoints_visibility = keypoints_visibility

    @property
    def keypoints(self):
        return self._keypoints

    @keypoints.setter
    def keypoints(self, kpts: List[float]):
        if kpts is None:
            return

        if not check_keypoints(kpts):
            raise ValueError("The provided list is probably not keypoints "
                             "because its length is not 51, 39, 34 or 26.")

        if has_head(kpts):
            kpts = king_of_france(kpts)

        if has_visibility(kpts):
            self._keypoints_visibility = kpts[2::3]
            del kpts[2::3]

        self._keypoints = tuple(kpts)

    @property
    def keypoints_visibility(self):
        return self._keypoints

    @keypoints_visibility.setter
    def keypoints_visibility(self, values: List[float]):
        if values is None:
            return

        if not len(values) in [17, 13]:
            raise ValueError("visibility keypoints are not the right length. "
                             "They should be of length 17 or 13")

        if len(values) == 17:
            values = values[4:]

        self._keypoints_visibility = tuple(values)

    @property
    def xy_keypoints(self) -> Tuple[Tuple, Tuple]:
        """
        return a tuple of x and y coordinates

        Returns
        -------
        Tuple[Tuple, Tuple]
            tuple of x, y coordinates ((x1, x2, ...), (y1, y2, ...))
        """
        return self.keypoints[::2], self.keypoints[1::2]

    @property
    def height(self) -> float:
        """return a skeleton height using the distance from the neck
        to the ankles"""
        if self._height is None:
            kp_neck = self.get_kpt(BodyKpt.Neck)

            kp_l_ankle = self.get_kpt(BodyKpt.LAnkle)
            kp_r_ankle = self.get_kpt(BodyKpt.RAnkle)
            kp_mid_ankle = middle_keypoint(kp_l_ankle, kp_r_ankle)

            self._height = ((kp_neck[0] - kp_mid_ankle[0])**2
                            + (kp_neck[1] - kp_mid_ankle[1])**2
                            )**0.5
        return self._height

    @property
    def width(self) -> float:
        """return the width of the bounding box"""
        _, _, width, _ = self.get_bbox("btwh", True)

        return width

    def bbox_setter(self, value: Sequence):
        """set the bounding box value after some basic verification."""
        if value == ():
            return

        if len(value) != 4:
            raise ValueError("Bounding box has more than 4 coordinates.")

        x_left, y_top, x_right, y_bottom = value

        if x_left > x_right or y_top < y_bottom:
            raise ValueError("The input bounding box should be bottom-left, top-right "
                             "coordinates, i.e. (x_left, y_top, x_right, y_bottom).")

        self._boundbing_box_bltr = value
    bbox = property(None, bbox_setter)

    @property
    def bbox_tlbr(self) -> List[float]:
        """
        return the top-left, bottom-right bounding box.

        Returns
        -------
        List[float]
            x_left, y_top, x_right, y_bottom
        """
        return self.get_bbox("tlbr")

    @property
    def bbox_tlwh(self) -> List[float]:
        """
        return the top-left, width-height bounding box.

        Returns
        -------
        List[float]
            x_left, y_top, width, height
        """
        return self.get_bbox("tlwh")

    @property
    def bbox_bltr(self) -> List[float]:
        """
        return the bottom-left, top-right bounding box.

        Returns
        -------
        List[float]
            x_left, y_bottom, x_right, y_top
        """
        return self._boundbing_box_bltr

    @property
    def bbox_blwh(self) -> List[float]:
        """
        return the bottom-left, width-height bounding box.

        Returns
        -------
        List[float]
            x_left, y_bottom, width, height
        """
        return self.get_bbox("blwh")

    def get_bbox(self, direction: str, allow_estimation: bool = True) -> List[float]:
        """
        return a bounding box in the correct format

        Parameters
        ----------
        direction : str
            "tlbr": top-left, bottom-right
            "tlwh": top-left, width-height
            "bltr": bottom-left, top-right
            "blwh": bottom-left, width-height
        allow_estimation: bool
            allow the bounding box to be computed from the keypoints in
            case the bounding box is not available.

        Returns
        -------
        List[float]
            bounding box
        """
        if self._boundbing_box_bltr is None:
            if not allow_estimation:
                raise AttributeError("There is no bounding box associated to this skeleton.")
            bbox = self._estimated_bbx()
        else:
            bbox = list(self._boundbing_box_bltr)

        x_left, y_top, x_right, y_bottom = bbox

        if direction.endswith("wh"):
            width = x_right - x_left
            height = y_top - y_bottom
            bbox[2:] = [width, height]

        if direction.startswith('tl'):
            bbox[:2] = [x_left, y_top]

        return bbox

    def _estimated_bbx(self) -> List[float]:
        """
        return an estimation of the bounding box from the min and max
        of the keypoints coordinates.

        Returns
        -------
        List[float]
            estimated bottom-left, top-right bounding box coordinates.
        """
        kpts_x, kpts_y = self.xy_keypoints

        return [min(kpts_x), min(kpts_y), max(kpts_x), max(kpts_y)]

    def get_kpt(self, kp_name: BodyKpt) -> Tuple[float, float]:
        """
        get the x, y coordinates of a keypoint

        Parameters
        ----------
        kp_name : BodyKpt
            name of the body keypoint e.g. BodyKpt.LAnkle

        Returns
        -------
        Tuple[float, float]
            x, y coordinates of the keypoint
        """
        return self.keypoints[2 * kp_name: 2 * kp_name + 2]

    def get_angles(self, angle_list: List[Tuple[BodyAngles]]) -> List[float]:
        """
        compute angles between 3 keypoints.

        Parameters
        ----------
        angle_list : list[tuple[int, int, int]]
            List of three-keypoint-indexes to compute angle for. You can use
            preexisting lists from the BodyKpt class. e.g. BodyKpt.BASIC_ANGLE_LIST or
            BodyKpt.MEDIUM_ANGLE_LIST. You can also use joints from the BodyKpt class
            e.g. [LKnee_angle, LElbow_angle, LShoulder_angle, LHip_angle].

        Example
        -------
        compute_angles(keypoints, [(BodyKpt.LHip, BodyKpt.LKnee, BodyKpt.LAnkle)])
        # will compute the 2D angle between (BodyKpt.LHip, BodyKpt.LKnee) and
        (BodyKpt.LHip, BodyKpt.LKnee).
        """
        angles = [0] * len(angle_list)

        for i, (angle) in enumerate(angle_list):
            if isinstance(angle, BodyAngles):
                kp_1, kp_2, kp_3 = angle.value
            else:
                kp_1, kp_2, kp_3 = angle

            kp_1 = self.get_kpt(kp_1)
            kp_2 = self.get_kpt(kp_2)
            kp_3 = self.get_kpt(kp_3)

            angles[i] = three_points_angle(kp_1, kp_2, kp_3)

        return angles

    def relative_to_neck(self) -> List[float]:
        """
        return the keypoints coordinates with the Neck as the origin
        """
        x_offset, y_offset = self.get_kpt(BodyKpt.Neck)

        keypoints = np.array(self._keypoints)
        keypoints[0::2] = keypoints[0::2] - x_offset
        keypoints[1::2] = keypoints[1::2] - y_offset

        return keypoints

    def toJson(self):
        """Serialise a skeleton to JSON."""
        return {"bbox_bltr": self._boundbing_box_bltr,
                "score": self._score,
                "keypoints": self._keypoints,
                "keypoints_visibility": self._keypoints_visibility,
                }


def check_keypoints(keypoints: List[float]) -> bool:
    """
    verify if the list can be keypoints.

    51: head and visibility
    39: head but no visibility
    34: no head but visibility
    26: no head and no visibility

    Parameters
    ----------
    keypoints : List[float]
        list of body keypoints to check.

    Returns
    -------
    bool
        whether or not they are valid keypoints.
    """
    if len(keypoints) in [51, 39, 34, 26]:
        return True
    return False


def has_head(keypoints: List[float]) -> bool:
    """
    verify is the keypoints have head keypoints.

    Parameters
    ----------
    keypoints : List[float]
        list of body keypoints to check.

    Returns
    -------
    bool
        whether or not the keypoints have head keypoints.
    """
    if len(keypoints) in [51, 34]:
        return True
    return False


def has_visibility(keypoints: List[float]) -> bool:
    """
    verify is the keypoints have visibility keypoints.

    Parameters
    ----------
    keypoints : List[float]
        list of body keypoints to check.

    Returns
    -------
    bool
        whether or not the keypoints have visibility keypoints.
    """
    if len(keypoints) in [51, 39]:
        return True
    return False


def round_list(list_to_round: list) -> list:
    """round all the values of a list. Useful to save a lot of space
    when saving the skeletons to a file"""
    return [round(value) for value in list_to_round]


def king_of_france(keypoints: list) -> list:
    """replace the head of a skeleton by its neck"""
    period = None
    if len(keypoints) == 51:
        period = 3
    elif len(keypoints) == 34:
        period = 2

    if period is not None:
        LEar_index = 3
        REar_index = 4
        kp_LEar = keypoints[LEar_index * period:(LEar_index + 1) * period]
        kp_REar = keypoints[REar_index * period:(REar_index + 1) * period]
        neck_kp = middle_keypoint(kp_LEar, kp_REar)
        return neck_kp + keypoints[5 * period:]

    raise ValueError("The skeleton is already beheaded")


def middle_keypoint(kp_1: Union[list, np.ndarray], kp_2: Union[list, np.ndarray]):
    """create a middle keypoint from two keypoint. Using numpy.mean was
    significantly slower."""
    new_kp = [0] * len(kp_1)
    for i in range(len(kp_1)):
        new_kp[i] = (kp_1[i] + kp_2[i]) / 2

    return new_kp


def three_points_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """
    compute an angle from 3 points.

    Parameters
    ----------
    p1, p2, p3 : Tuple[float, float]
        (x, y) coordinates of the point.

    Returns
    -------
    float
        angle between (p1, p2) and (p2, p3)
    """
    if np.allclose(p1, p2) or np.allclose(p2, p3) or np.allclose(p1, p3):
        return 0

    ba = tuples_substract(p1, p2)
    bc = tuples_substract(p3, p2)

    cosine_angle = (np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))).astype(np.float16)
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def tuples_substract(a: Sequence, b: Sequence) -> List:
    """return the element-by-element a - b operation"""
    ba = [0] * len(a)

    for i in range(len(a)):
        ba[i] = a[i] - b[i]

    return ba