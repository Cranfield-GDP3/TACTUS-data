from pathlib import Path
import os
import cv2


def extract_frames(
        video_path: Path,
        output_dir: Path,
        desired_fps: int):
    """
    Extract frames from a video source at a specific frame rate.
    Extracted frames will be name with leading 0 (e.g. '0004.jpg').
    Parameters
    ----------
    video_path : Path
        The video path that gives the video source
    output_dir : Path
        The location where the frame images are saved
    desired_fps : int
        The fps we want to have
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    extract_frequency = int(fps / desired_fps)

    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frame_len = len(str(n_frame))


    count = 0
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        if desired_fps == -1:
            save_path = str(result_path / f"{str(count).zfill(n_frame_len)}.jpg")
            cv2.imwrite(save_path, frame)
        else:
            if count % extract_frequency == 0:
                frame_name = str(count).zfill(n_frame_len)
                save_path = output_dir.absolute() / f"{frame_name}.jpg"
                cv2.imwrite(str(save_path), frame)

        count += 1

    cap.release()
