"""hanldes operations relative to the UT interaction dataset.
https://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html"""

from pathlib import Path

import zipfile
import io
import shutil

import requests
from tactus_data.utils.read_dataset_urls import read_dataset_urls


class UTInteraction:
    DEFAULT_RAW_DIR = Path("data/raw/ut_interaction")

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

        self.move_videos(download_dir)
        shutil.rmtree(download_dir / "segmented_set1")
        shutil.rmtree(download_dir / "segmented_set2")

        self.create_labels(download_dir)

    def move_videos(self, download_dir: Path):
        """
        The archive contains two subfolders and this function move every
        video from the subfolders to the parent folder.

        Parameters
        ----------
        download_dir : Path, optional
            The path where the data have been downloaded, by default DEFAULT_RAW_DIR
        """

        for video_path in download_dir.glob("*/*.avi"):
            video_path.rename(download_dir / video_path.name)

    ACTION_INDEXES = ["hand shaking", "hugging", "kicking",
                      "pointing", "punching", "pushing"]


    def create_labels(self, download_dir: Path):
        """
        generate a label file from the name of the video which contains:
        a sample number, a sequence number, and the action number.

        Parameters
        ----------
        download_dir : Path, optional
            The path where the data have been downloaded, by default DEFAULT_RAW_DIR
        """

        for video_path in download_dir.glob("*.avi"):
            _, _, action = video_path.stem.split("_")
            corresponding_action = self.ACTION_INDEXES[int(action)]

            label_path = download_dir / (video_path.stem + ".txt")
            with label_path.open("w", encoding="utf8") as label_file:
                label_file.write(corresponding_action)
