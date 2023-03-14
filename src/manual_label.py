import time
from pathlib import Path
import json
import cv2
import keyboard
import os


def _change_frame(current_frame: int,
                  frame_vid: list,
                  video_position: list,
                  text: str):

    #Need English keyboard
    right = "b"
    left = "c"
    enter = "enter"
    cv2.imshow(text, frame_vid[current_frame])
    cv2.moveWindow(text, video_position[0], video_position[1])
    cv2.setWindowProperty(text, cv2.WND_PROP_TOPMOST, 1)

    while True:
        cv2.imshow(text, frame_vid[current_frame])

        print("\nCurrent frame :", current_frame,
              "\nIf" +text+" press : Enter\nElse press the arrow <-c b-> to change frame")
        cv2.waitKey(30)
        key_press = keyboard.read_key()
        #time.sleep(0.05)
        if key_press == enter:
            time.sleep(1)
            break
        elif key_press == left and current_frame > 0:
            current_frame -= 1
        elif key_press == right and current_frame < (
                len(frame_vid) - 2):  # -1 because 0 indexed and another -1 because last value is none with imrea
            current_frame += 1

    cv2.destroyAllWindows()
    return current_frame


def label_video_frame(path_file: Path,
                      path_output: Path,
                      show_vid: bool = True,
                      ):
    """
    Review all videos (.mp4 / .avi) of a folder to generate *data.json files in output folder

    Parameters
    ----------
    path_file : Path, path where the original json files are located
    path_output : Path, path where the new generated data are saved inside of the folder of the corresponding video name
    show_vid : bool, flag to enable video preview before manual classification
    """

    # Temp code to identify label for ut-interaction
    ACTION_INDEXES = ["neutral", "neutral", "kicking",
                      "neutral", "punching", "pushing"]
    # Temps code

    video_position = [500,500]
    list_files = Path.iterdir(path_file)
    list_vid = []
    for file in list_files:
        if file.suffix == '.mp4' or file.suffix == '.avi':
            list_vid.append(file)

    for vid_path in list_vid:
        frame_vid = []
        vid = cv2.VideoCapture(str(vid_path))

        # Temp code to identify label for ut-interaction
        _, _, action = vid_path.stem.split("_")
        label = ACTION_INDEXES[int(action)]
        # Temp code

        while vid.isOpened():
            ret, frame = vid.read()
            frame_vid.append(frame)
            if ret == True:
                # Display the resulting frame
                resolution = list(frame.shape[0:2])
                if show_vid==True:
                    cv2.imshow('Video', frame)
                    cv2.moveWindow('Video',video_position[0],video_position[1])
                    cv2.setWindowProperty('Video', cv2.WND_PROP_TOPMOST, 1)
                    if cv2.waitKey(25) & 0xFF == ord('q'): #not working on pycharm
                        break
            else:
                break
        vid.release()
        cv2.destroyAllWindows()
        action_left = 1
        start_frame =-1
        end_frame =-1
        current_frame=0
        data_dic = {}
        data_dic["resolution"] = resolution
        data_dic["classes"] = []

        while action_left == 1:
            start_frame = _change_frame(current_frame,frame_vid,video_position,"Start Frame")
            end_frame = _change_frame(start_frame, frame_vid, video_position, "End Frame")
            data_dic["classes"].append({"classification" : label,
                                        "start_frame": start_frame,
                                        "end_frame": end_frame})
            print("Are they any other action in the video ?\n Press any key to continue or Enter to go to the next video")
            key_press = keyboard.read_key()
            if key_press == "enter":
                action_left=0

        with open(str(path_output) + '\\' + vid_path.stem +'\\' + vid_path.stem + '.label.json', "w") as outfile:
            json.dump(data_dic,
                      outfile)

def mkdir(raw_dir: Path,
          target_dir: Path):
    """Automatically generate a corresponding video folder under
    the ut_interaction folder

    Parameters
    ----------
    raw_dir : Path, path where the raw video files are located
    target_dir : Path, path where the folder with the corresponding video name should be saved
    """

    listDir = os.listdir(raw_dir)
    for dir in listDir:

        if os.path.isdir(dir) or 'new.py' == dir:
            continue
        dirName = os.path.splitext(dir)[0]
        dirName = target_dir + '/' + dirName
        if not os.path.exists(dirName):
            os.mkdir(dirName)
