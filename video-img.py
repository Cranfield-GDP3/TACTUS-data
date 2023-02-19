import cv2
import time
import numpy as np
import os



def save_img(video_path, result_path, interval_num, resize_wh, rot90=False, show=False):
    '''
    将视频每帧保存到文件夹，命名格式：视频名_时间戳_num
    Save each frame of the video to a folder, naming format: video name_timestamp_num
    :param video_path: 视频路径
    :param result_path: 结果保存路径
    :param interval_num: 每间隔几帧保存一帧 (Save a frame every few frames)
    :param resize_wh: 指定保存图片的长宽 (Specifies the length and width of the saved image)
    :param rot90: 是否90度旋转图片 (Whether to rotate the picture 90 degrees)
    :param show: 是否进行 窗口展示图片 (Whether to display the picture in the window)
    :return:
    '''
    i = 0
    cap = cv2.VideoCapture(video_path)
    # 视频长宽 （Video length and width）
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    while(True):
        s_time = int(time.time())
        i += 1
        ret, frame = cap.read()
        if i % interval_num != 0:
            continue
        # 横屏转竖屏 （Landscape to portrait）
        if rot90:
            frame = np.rot90(frame)
        milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
        # transfer to s
        # s_time = round(milliseconds / 1000, 2)
        # resize
        if resize_wh != False:
            if w != resize_wh[0] or h != resize_wh[1]:
                frame = cv2.resize(frame, resize_wh)
        file_path = os.path.join(result_path, name + '/' + name + '_' + str(s_time) + '_' + str(int(i)) + '.jpg')
        cv2.imencode('.jpg', frame)[1].tofile(file_path)
        # 展示窗口 （display window）
        if show:
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    # 视频名 （video name）
    video_path = r'nofi001.mp4'
    # 保存路径 （saved image path）
    result_path = 'D:/GDP/TESTING/vi-im/'
    # 间隔interval_num取一帧，默认为1，一帧不跳，跳1帧为2 (How many frames to take an image at intervals)
    interval_num = 5
    # resize_wh = (1920, 1080)
    resize_wh = False
    rot90 = False
    show = False
    name = video_path.split('\\')[-1].split('.')[0]
    if not os.path.exists(os.path.join(result_path, name)):
        os.makedirs(os.path.join(result_path, name))
    # 视频到图片 (video to image)
    save_img(video_path, result_path, interval_num, resize_wh, rot90, show)