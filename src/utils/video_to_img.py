import cv2
from pathlib import Path


def extract_frames(
        video_path: Path,
        result_path: Path,
        desired_fps: int):
    """Return the series of frames that convert from a
    video source.

    Parameters
    ----------
    video_path : Path
        The video path that gives the video source
    result_path : Path
        The location where the frame image is stored
    desired_fps : int
        The fps we want to have
    """

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    extract_frequency = int(fps / desired_fps)
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frame_len = len(str(n_frame))

    count = 1
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        if count % extract_frequency == 0:
            save_path = result_path / f"{str(count).zfill(n_frame_len)}.jpg"
            cv2.imwrite(save_path, frame)
        count += 1

    cap.release()
