from pathlib import Path
import cv2
from tactus_yolov7 import Yolov7

MODEL_WEIGHTS_PATH = Path("data/raw/model/yolov7-w6-pose.pt")


def yolov7(input_dir: Path, model: Yolov7):
    """extract skeleton keypoints with yolov7. It extract skeletons
    from every *.jpg image in the input_dir"""
    formatted_json = {}
    formatted_json["frames"] = []

    min_nbr_skeletons = float("inf")
    max_nbr_skeletons = 0
    for frame_path in Path(input_dir).glob("*.jpg"):
        frame_json = {"frame_id": frame_path.name}

        img = cv2.imread(str(frame_path))
        skeletons = model.predict_frame(img)

        min_nbr_skeletons = min(min_nbr_skeletons, len(skeletons))
        max_nbr_skeletons = max(max_nbr_skeletons, len(skeletons))

        frame_json["skeletons"] = skeletons
        formatted_json["frames"].append(frame_json)

    formatted_json["min_nbr_skeletons"] = min_nbr_skeletons
    formatted_json["max_nbr_skeletons"] = max_nbr_skeletons

    return formatted_json

def delete_confidence_kpt(skeleton: list) -> list:
    """delete the confidence for each keypoint of a skeleton"""
    return drop_every_nth_index(skeleton, 3)

def drop_every_nth_index(initial_list: list, n: int) -> list:
    """remove every nth index from a list"""
    del initial_list[n-1::n]
    return initial_list

def round_values(skeleton: list) -> list:
    """round all the values of a list. Useful to save a lot of space
    when saving the skeletons to a file"""
    return [round(kpt) for kpt in skeleton]
