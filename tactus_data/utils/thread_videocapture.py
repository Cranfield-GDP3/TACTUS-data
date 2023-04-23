"""
A videoCapture API extension that allows for subsampling and threading
"""
from collections import deque
from pathlib import Path
from typing import Union, Literal
from time import time
import threading
import warnings

import cv2


class VideoCapture:
    """
    extends [cv2::VideoCapture class](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html)
    for video or stream subsampling.

    Parameters
    ----------
    filename : Union[str, int]
        Open video file or image file sequence or a capturing device
        or a IP video stream for video capturing.
    target_fps : int, optional
        the target frame rate. To ensure a constant time period between
        each subsampled frames, this parameter is used to compute a
        integer denominator for the extraction frequency. For instance,
        if the original capture is 64fps and you want a 30fps capture
        out, it is going to take one frame over two giving an effective
        frame rate of 32fps.
        If None, will extract every frame of the capture.
    capture_fps : float, optional
        the input frame rate. Can be specified to avoid the frame rate
        estimation part as it can be imprecise or couldn't work if the
        stream does not emit any frame for now.
    drop_warning_enable : bool, optional
        whether or not to show a warning if a frame had to be dropped.
        When the buffer is full and a new frame is coming in, a frame
        has to be dropped in order to make space in the buffer.
        By default True.
    """
    def __init__(self,
                 filename: Union[Path, str, int],
                 target_fps: int = None,
                 capture_fps: float = None,
                 drop_warning_enable: bool = True
                 ) -> None:
        _filename = filename
        if isinstance(filename, Path):
            _filename = str(filename)
        self._cap = cv2.VideoCapture(_filename)
        self.cap_name = filename
        self.mode = self.get_cap_mode(filename)

        self.target_fps = target_fps
        self._capture_fps = self.get_capture_fps(capture_fps)
        self.extract_freq = self.get_extract_frequency()
        self._out_fps = self._capture_fps / self.extract_freq

        self._imgs_queue = Queue(maxlen=5)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._thread_read)
        self._thread.start()

        self.drop_warning_enable = drop_warning_enable

    def get_capture_fps(self, value: Union[None, float]) -> float:
        """
        return the input capture frame rate, either from its property
        or via an estimation, or the user input if provided.

        Parameters
        ----------
        value : Union[None, float]
            user input

        Returns
        -------
        float
            the capture frame rate.
        """
        if value is None:
            # cv2.CAP_PROP_FPS returns 0 if the property doesn't exist
            capture_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if capture_fps == 0:
                capture_fps = self.estimate_capture_fps()
        else:
            capture_fps = value

        return capture_fps

    def estimate_capture_fps(self, evaluation_period: int = 5):
        """evaluate the frame rate over a period of 5 seconds"""
        frame_count = 0
        while self.isOpened():
            ret, _ = self._cap.read()
            if ret is True:
                if frame_count == 0:
                    start = time()

                frame_count += 1

                if time() - start > evaluation_period:
                    break

        if frame_count == 0:
            raise FileNotFoundError("Could not estimate the input capture fps. You can specify "
                                    "the input frame rate using the `capture_fps` argument")

        return round(frame_count / (time() - start), 2)

    def get_extract_frequency(self):
        """evaluate the frame rate over a period of 5 seconds"""
        if self.target_fps is None:
            return 1

        extract_freq = int(self.capture_fps / self.target_fps)

        if extract_freq == 0:
            raise ValueError("desired_fps is higher than half the stream frame rate")

        return extract_freq

    def get_cap_mode(self, filename: Union[Path, str, int]) -> Literal["stream", "video"]:
        """
        tries to identify what is the provided file.

        Parameters
        ----------
        filename : Union[Path, str, int]

        Returns
        -------
        str
            name of the capture mode, either "stream" or "video".
        """
        if isinstance(filename, (int, float)):
            return "stream"

        filename = Path(filename)
        if filename.is_file():
            return "video"

        return "stream"

    def isOpened(self):
        """Returns true if video capturing has been initialized already."""
        return self._cap.isOpened()

    def release(self):
        """Closes video file or capturing device."""
        self._stop_event.set()
        self._cap.release()

    def read(self):
        """
        Grabs, decodes and returns the next subsampled video frame.
        If there is no image in the queue, wait for one to arrive.
        """
        if self._stop_event.isSet() and self._imgs_queue.is_empty():
            return None

        while self._imgs_queue.is_empty():
            continue

        return self._imgs_queue.popleft()

    def _thread_read(self):
        frame_count = 0
        while not self._stop_event.isSet():
            ret, frame = self._cap.read()
            # stop thread when a video is over
            if self.mode == "video" and ret is False:
                self._stop_event.set()
                continue

            if ret is False:
                continue

            frame_count += 1
            if frame_count != self.extract_freq:
                continue

            frame_count = 0
            if self.mode == "stream":
                # this avoid the queue shrinking before the result of
                # full() can be used.
                with self._imgs_queue.lock:
                    if self._imgs_queue.maxlen <= len(self._imgs_queue):
                        self._imgs_queue.popleft()

                        if self.drop_warning_enable:
                            warnings.warn("frame dropped")
            if self.mode == "video":
                # when dealing with a video, we can wait for the queue
                # being not full
                while self._imgs_queue.is_full():
                    continue

            # we don't need the lock as this is the only thread that
            # can put things in the queue
            self._imgs_queue.append(frame)

    def __del__(self):
        if self.isOpened():
            self.release()


class Queue(deque):
    def __init__(self, maxlen: int):
        self.lock = threading.Lock()
        super().__init__(maxlen=maxlen)

    def is_empty(self):
        return len(self) == 0

    def is_full(self):
        return len(self) == self.maxlen
