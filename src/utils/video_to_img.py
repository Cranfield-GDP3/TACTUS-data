import cv2
from pathlib import Path


def extract_frames(video_path: Path, result_path: Path, desired_fps: int):
    '''
        :param video_path: video path
        :param result_path: the location where the frame image is stored
        :param desired_fps
    '''

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    extract_frequency = int(fps / desired_fps)
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = len(str(n_frame))

    count = 1
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        if count % extract_frequency == 0:
            save_path = result_path / f"{str(count).zfill(n)}.jpg"
            cv2.imwrite(save_path, frame)
        count += 1
    cap.release()
