"""hanldes operations relative to the UT interaction dataset.
https://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html"""

from pathlib import Path

import zipfile
import io
import shutil

import requests
from tactus_data.utils.read_dataset_urls import read_dataset_urls
from tactus_data.utils.interim_name_convention import interim_name_convention
from tactus_data.utils.video_to_img import extract_frames as video_to_images
from tactus_data.utils.alphapose import alphapose_skeletonisation


class UTInteraction:
    NAME = "ut_interaction"
    DEFAULT_RAW_DIR = Path(f"data/raw/{NAME}")
    DEFAULT_INTERIM_DIR = Path("data/interim/")
    DEFAULT_PROCESSED_DIR = Path("data/processed/")

    def download(self, download_dir: Path = DEFAULT_RAW_DIR):
        """
        Download and extract dataset from source.

        Parameters
        ----------
        download_dir : Path, optional
            The path where to download the data, by default DEFAULT_RAW_DIR
        """

        zip_file_urls = read_dataset_urls(key="UTInteraction")

        for zip_file_url in zip_file_urls:
            response = requests.get(zip_file_url, timeout=1000)
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_response:
                zip_response.extractall(download_dir)

        self._move_videos(download_dir)

    def extract_frames(
            self,
            download_dir: Path = DEFAULT_RAW_DIR,
            output_dir: Path = DEFAULT_INTERIM_DIR,
            desired_fps: int = 10
        ):
        """
        Extract frame for this dataset and label them by
        moving them into different subfolders.

        Parameters
        ----------
        download_dir : Path, optional
            The path where the data have been downloaded,
            by default DEFAULT_RAW_DIR
        output_dir : Path, optional
            The path to where to save the frames,
            by default DEFAULT_INTERIM_DIR
        desired_fps : int, optional
            The fps we want to have, by default 10
        """
        for video_path in download_dir.glob("*.avi"):
            label = self._label_from_path(video_path)
            uid = self._uid_from_path(video_path)

            frame_output_dir = (output_dir
                                / label
                                / interim_name_convention(self.NAME, uid, desired_fps))
            video_to_images(video_path, frame_output_dir, desired_fps)

    def extract_skeletons(
            self,
            interim_dir: Path = DEFAULT_INTERIM_DIR,
            output_dir: Path = DEFAULT_PROCESSED_DIR
        ):
        for extracted_frames_dir in interim_dir.glob(f"*/{self.NAME} *"):
            output_filename = f"{extracted_frames_dir.stem}.json"
            alphapose_skeletonisation(extracted_frames_dir.absolute(), output_dir.absolute() / output_filename)

    ACTION_INDEXES = ["neutral", "neutral", "kicking",
                      "neutral", "punching", "pushing"]

    def _move_videos(self, download_dir: Path):
        """
        The archive contains two subfolders and this function move every
        video from the subfolders to the parent folder.

        Parameters
        ----------
        download_dir : Path
            The path where the data have been downloaded
        """

        for video_path in download_dir.glob("*/*.avi"):
            video_path.rename(download_dir / video_path.name)

        shutil.rmtree(download_dir / "segmented_set1")
        shutil.rmtree(download_dir / "segmented_set2")

    def _label_from_path(self, video_path: Path) -> str:
        """
        Extract the label from the video name.

        Parameters
        ----------
        video_path : Path
            The path of the video to extract a label from.

        Returns
        -------
        str :
            the corresponding label
        """
        _, _, action = video_path.stem.split("_")
        label = self.ACTION_INDEXES[int(action)]

        return label

    def _uid_from_path(self, video_path: Path) -> str:
        """
        Compute an unique id from the path.

        Parameters
        ----------
        video_path : Path
            The path of the video to extract a label from.

        Returns
        -------
        str :
            unique id for the video in the dataset
        """
        sample_number, sequence_number, _ = video_path.stem.split("_")
        unique_id = f"{sample_number.zfill(2)}_{sequence_number}"

        return unique_id
