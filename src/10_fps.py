import cv2
import os
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
        if desired_fps == -1:
            save_path = str(result_path / f"{str(count).zfill(n_frame_len)}.jpg")
            cv2.imwrite(save_path, frame)
        else:
            if count % extract_frequency == 0:
                save_path = str(result_path / f"{str(count).zfill(n_frame_len)}.jpg")
                cv2.imwrite(save_path, frame)

        count += 1

    cap.release()

def mkdir(
        raw_dir: Path,
        target_dir: Path):
    """Automatically generate a corresponding video folder
    under the ut_interaction folder

    Parameters
    ----------
    raw_dir : Path
        The path where the raw video files are located
    target_dir : Path
        The path where the folder with the corresponding video name should be saved
    """

    listDir = os.listdir(raw_dir)
    for dir in listDir:

        if os.path.isdir(dir) or 'new.py' == dir:
            continue
        dirName = os.path.splitext(dir)[0]
        dirName = target_dir + '/' + dirName
        if not os.path.exists(dirName):
            os.mkdir(dirName)

# 读取视频路径名称
def get_videoNames(
        rootdir: Path):
    """Load the path of the video.

    Parameters
    ----------
    rootdir : Path
        The video path that gives the video source
    """

    fs = []
    for root, dirs, files in os.walk(rootdir,topdown = True):
        for name in files:
            _, ending = os.path.splitext(name)
            if ending == ".avi":
                    fs.append(os.path.join(root,name))
    return fs

def ten_fps (
        video_path: Path,
        result_path: Path):
    """Automatically generate a corresponding frames of video
    and put them under the 10fps folder

    Parameters
    ----------
    video_path : Path
        The video path that gives the video source
    result_path : Path
        The location where the frame image is stored
    """

    mkdir(video_path, result_path)
    list = get_videoNames(video_path)
    for i in list:
        video_name = os.path.basename(i)
        file_name = video_name.split('.')[0]
        fps_path = Path(str(result_path) + '/' + file_name)
        ten_fps = Path(str(fps_path) + '/' + '10fps')
        if not os.path.exists(ten_fps):
            os.mkdir(ten_fps)
        extract_frames(i, ten_fps, 10)
