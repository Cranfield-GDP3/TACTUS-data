import cv2
import shutil
from pathlib import Path

# global variable
VIDEO_PATH = Path("D:/GDP/TESTING/nofi001.mp4")
EXTRACT_FOLDER = Path("D:/GDP/TESTING/vi-im/")

def extract_frames(video_path, result_path, index, desired_fps):
    '''
        :param video_path: video path
        :param result_path: the location where the frame image is stored
        :param index: the starting sequence number of the frame save filename,
                      this sequence number will be appended to the filename of each saved frame image
        :param desired_fps
        '''

    video = cv2.VideoCapture() # read frames from video source
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) # get the frame rate of the video
    EXTRACT_FREQUENCY = int(fps / desired_fps) # Frame extraction frequency
    print("EXTRACT_FREQUENCY: ", EXTRACT_FREQUENCY)
    if not video.open(str(video_path)):
        print("can not open the video")
        exit(1)
    count = 1
    while True:
        _, frame = video.read()
        if frame is None:
            break
        if count % EXTRACT_FREQUENCY == 0:
            # Determine whether the current frame needs to be extracted
            save_path = "{}/{:>03d}.jpg".format(result_path, index)
            # save the current frame as an image file
            cv2.imwrite(save_path, frame)
            index += 1
        count += 1
    video.release()
    print("Original fps: ", fps)
    print("Totally save {:d} images".format(index-1))


def main():
    # recursively delete the folder where the frame images was stored before,
    # and create a new one
    try:
        shutil.rmtree(EXTRACT_FOLDER)
    except OSError:
        pass
    import os
    os.mkdir(EXTRACT_FOLDER)
    # Extract the frames and save it to the specified path
    extract_frames(VIDEO_PATH, EXTRACT_FOLDER, 1, 25)