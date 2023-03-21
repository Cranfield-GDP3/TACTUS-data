from pathlib import Path
import cv2
from tactus_yolov7 import Yolov7

MODEL_WEIGHTS_PATH = Path("data/raw/model/yolov7-w6-pose.pt")


def yolov7(input_dir: Path, model: Yolov7):
    formatted_json = {}
    formatted_json["frames"] = []
    for frame_path in Path(input_dir).glob("*.jpg"):
        frame_json = {"frame_id": frame_path.name}

        img = cv2.imread(str(frame_path))
        skeletons = model.predict_frame(img)

        frame_json["skeletons"] = skeletons
        formatted_json["frames"].append(frame_json)

    return formatted_json
