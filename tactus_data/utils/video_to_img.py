from pathlib import Path
import cv2


def extract_frames(
        video_path: Path,
        output_dir: Path,
        fps: int = None):
    """
    Extract frames from a video source at a specific frame rate.
    Extracted frames will be name with leading 0 (e.g. '0004.jpg').

    Parameters
    ----------
    video_path : Path
        The video path that gives the video source
    output_dir : Path
        The location where the frame images are saved
    fps : int, optional
        The fps we want to have. Extract every frame if None, by default None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None:
        extract_frequency = 1
    else:
        extract_frequency = int(fps / fps)

    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frame_len = len(str(n_frame))

    count = 0
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        if count % extract_frequency == 0:
            frame_name = str(count).zfill(n_frame_len)
            save_path = output_dir.absolute() / f"{frame_name}.jpg"
            cv2.imwrite(str(save_path), frame)

        count += 1

    cap.release()
