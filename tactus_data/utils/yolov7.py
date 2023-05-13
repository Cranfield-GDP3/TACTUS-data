# WIP
from typing import List
from pathlib import Path

import numpy as np
import tactus_yolov7
from tactus_yolov7.utils.datasets import letterbox
from ultralytics.yolo.utils.ops import scale_boxes, scale_coords

from tactus_data.utils.skeleton import Skeleton


class Yolov7(tactus_yolov7.Yolov7):
    """custom interface to the yolov8 models"""
    def __init__(self, model_dir: Path, model_name: str, device: str, half: bool = True):
        super().__init__(model_dir / model_name, device=device)

    def __call__(self, img: np.ndarray) -> List[Skeleton]:
        return self.predict(img)

    def predict(self, img: np.ndarray) -> List[Skeleton]:
        img0_size = img.shape[:2]
        image = letterbox(img, (960, 960), stride=64, auto=True)[0]
        img1_size = image.shape[:2]

        skeletons = super().predict_frame(image)

        skeleton_objects = [{}] * len(skeletons)
        for i, skeleton in enumerate(skeletons):
            box = np.array(skeleton["box"])
            cx, cy, width, height = scale_boxes(img1_size, box, img0_size)
            left = cx - width / 2
            bottom = cy - height / 2
            right = cx + width / 2
            top = cy + height / 2
            skeleton_objects[i] = Skeleton((left, bottom, right, top), keypoints=skeleton["keypoints"], score=skeleton["score"])

        return skeleton_objects
