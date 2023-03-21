from pathlib import Path
from torchvision import transforms
import cv2
from torch.hub import download_url_to_file
from tactus_yolov7 import Yolov7

MODEL_WEIGHTS_PATH = Path("data/raw/model/yolov7-w6-pose.pt")


def yolov7(input_dir: Path):
    model = Yolov7(MODEL_WEIGHTS_PATH)

    formatted_json = {}
    formatted_json["frames"] = []
    for frame_path in Path(input_dir).glob("*.jpg"):
        frame_json = {"frame_id": frame_path.name}

        img = cv2.imread(str(frame_path))
        skeletons = model.predict_frame(img)

        frame_json["skeletons"] = skeletons
        formatted_json["frames"].append(frame_json)

    return formatted_json

def download_weights():
    """Download yolov7 pose weights"""

    url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt"
    download_url_to_file(url, MODEL_WEIGHTS_PATH)
