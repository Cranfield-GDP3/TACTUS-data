from typing import List, Tuple, Union, Sequence
from enum import IntEnum, Enum
from json import JSONEncoder
import numpy as np


# allow custom serialization function for the Skeleton class
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default


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
    (BodyKpt.LShoulder, BodyKpt.LElbow),
    (BodyKpt.RElbow, BodyKpt.RWrist),
    (BodyKpt.LElbow, BodyKpt.LWrist),
    (BodyKpt.RShoulder, BodyKpt.Neck),
    (BodyKpt.LShoulder, BodyKpt.Neck),
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
    def __init__(self, bbox_lbrt: Sequence = (), score: float = None, keypoints: Sequence = None, keypoints_visibility: Sequence = None, tracking_id: int = None) -> None:
        self._boundbing_box_lbrt: Tuple[float, float, float, float] = None
        self._score: float = score
        self._keypoints: Tuple[float] = None
        self._keypoints_visibility: Tuple[bool] = None
        self._height: float = None
        self.tracking_id = tracking_id

        self.bbox = bbox_lbrt
        self.keypoints = keypoints
        self.keypoints_visibility = keypoints_visibility

    @property
    def keypoints(self):
        return self._keypoints

    @keypoints.setter
    def keypoints(self, kpts: List[List[float]]):
        if kpts is None:
            return

        # accept one dim list as input
        if len(kpts) in (26, 34):
            kpts = list(np.array(kpts).reshape((-1, 2)))
        if len(kpts) in (29, 51):
            kpts = list(np.array(kpts).reshape((-1, 3)))

        if not check_keypoints(kpts):
            raise ValueError("The provided list is probably not keypoints because "
                             "its length is not 17 nor 14, or each keypoints "
                             "have less than 2 or more than 3 coordinates.")

        if has_head(kpts):
            kpts = king_of_france(kpts)

        if has_visibility(kpts):
            self._keypoints_visibility = kpts[:][2]
            for i in range(len(kpts)):
                del kpts[i][2]

        self._keypoints = tuple(kpts)

    @property
    def keypoints_visibility(self):
        return self._keypoints_visibility

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
        return [xy[0] for xy in self.keypoints], [xy[1] for xy in self.keypoints]

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

    @property
    def score(self) -> float:
        """return the skeleton score"""
        return self._score

    def bbox_setter(self, value: Sequence):
        """set the bounding box value after some basic verification."""
        if value == ():
            return

        if len(value) != 4:
            raise ValueError("Bounding box has more than 4 coordinates.")

        x_left, y_bottom, x_right, y_top = value

        if x_left > x_right:
            x_left, x_right = x_right, x_left

        if y_bottom > y_top:
            y_bottom, y_top = y_top, y_bottom

        self._boundbing_box_lbrt = value
    bbox = property(None, bbox_setter)

    @property
    def bbox_ltrb(self) -> List[float]:
        """
        return the left-top, right-bottom bounding box.

        Returns
        -------
        List[float]
            x_left, y_top, x_right, y_bottom
        """
        return self.get_bbox("ltrb")

    @property
    def bbox_ltwh(self) -> List[float]:
        """
        return the left-top, width-height bounding box.

        Returns
        -------
        List[float]
            x_left, y_top, width, height
        """
        return self.get_bbox("ltwh")

    @property
    def bbox_lbrt(self) -> List[float]:
        """
        return the left-bottom, right-top bounding box.

        Returns
        -------
        List[float]
            x_left, y_bottom, x_right, y_top
        """
        return self._boundbing_box_lbrt

    @property
    def bbox_lbwh(self) -> List[float]:
        """
        return the left-bottom, width-height bounding box.

        Returns
        -------
        List[float]
            x_left, y_bottom, width, height
        """
        return self.get_bbox("lbwh")

    @property
    def bbox_cxcywh(self) -> List[float]:
        """
        return the center-x center-y, width-height bounding box.

        Returns
        -------
        List[float]
            x_center, y_center, width, height
        """
        return self.get_bbox("cxcywh")

    def get_bbox(self, direction: str, allow_estimation: bool = True) -> List[float]:
        """
        return a bounding box in the correct format

        Parameters
        ----------
        direction : str
            "ltrb": left-top, right-bottom
            "ltwh": left-top, width-height
            "lbrt": left-bottom, right-top
            "lbwh": left-bottom, width-height
            "cxcywh": center-x center-y, width-height
        allow_estimation: bool
            allow the bounding box to be computed from the keypoints in
            case the bounding box is not available.

        Returns
        -------
        List[float]
            bounding box
        """
        if self._boundbing_box_lbrt is None:
            if not allow_estimation:
                raise AttributeError("There is no bounding box associated to this skeleton.")
            bbox = self._estimated_bbx()
        else:
            bbox = list(self._boundbing_box_lbrt)

        x_left, y_bottom, x_right, y_top = bbox

        if direction.endswith("wh"):
            width = x_right - x_left
            height = y_top - y_bottom
            bbox[2:] = [width, height]

        if direction.endswith('rb'):
            bbox[:2] = [x_right, y_bottom]

        if direction.startswith('lt'):
            bbox[:2] = [x_left, y_top]

        if direction.startswith('cxcy'):
            bbox[:2] = [(x_left + x_right) / 2,
                        (y_top + y_bottom) / 2]

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
        return self.keypoints[kp_name]

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

    def relative_to_neck(self) -> np.ndarray:
        """
        return the keypoints coordinates with the Neck as the origin
        """
        x_offset, y_offset = self.get_kpt(BodyKpt.Neck)

        keypoints = np.array(self._keypoints)
        keypoints[:][0] = keypoints[:][0] - x_offset
        keypoints[:][1] = keypoints[:][1] - y_offset

        return keypoints

    def to_json(self):
        """Serialise a skeleton to JSON."""
        return {"bbox_lbrt": self.bbox_lbrt,
                "score": self._score,
                "keypoints": self.keypoints,
                "keypoints_visibility": self.keypoints_visibility,
                "tracking_id": self.tracking_id,
                }


def check_keypoints(keypoints: List[List[float]]) -> bool:
    """
    verify if the list can be keypoints.

    Parameters
    ----------
    keypoints : List[List[float]]
        list of body keypoints to check.

    Returns
    -------
    bool
        whether or not they are valid keypoints.
    """
    if len(keypoints) in (17, 14):
        return True
    if len(keypoints[0]) in (2, 3):
        return True
    return False


def has_head(keypoints: List[List[float]]) -> bool:
    """
    verify is the keypoints have head keypoints.

    Parameters
    ----------
    keypoints : List[List[float]]
        list of body keypoints to check.

    Returns
    -------
    bool
        whether or not the keypoints have head keypoints.
    """
    if len(keypoints) == 17:
        return True
    return False


def has_visibility(keypoints: List[List[float]]) -> bool:
    """
    verify is the keypoints have visibility keypoints.

    Parameters
    ----------
    keypoints : List[List[float]]
        list of body keypoints to check.

    Returns
    -------
    bool
        whether or not the keypoints have visibility keypoints.
    """

    if len(keypoints[0]) == 3:
        return True
    return False


def round_list(list_to_round: list) -> list:
    """round all the values of a list. Useful to save a lot of space
    when saving the skeletons to a file"""
    return [round(value) for value in list_to_round]


def king_of_france(keypoints: List[List[float]]) -> List[List[float]]:
    """replace the head of a skeleton by its neck"""

    if len(keypoints) == 17:
        LEar_index = 3
        REar_index = 4
        kp_LEar = keypoints[LEar_index]
        kp_REar = keypoints[REar_index]
        neck_kp = middle_keypoint(kp_LEar, kp_REar)
        return [neck_kp] + keypoints[5:]

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
