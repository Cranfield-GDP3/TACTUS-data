"""hanldes operations relative to the UT dataset.
https://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html"""

from typing import Union, List
from pathlib import Path
import zipfile
import io
import shutil
import requests
import yaml


class UTDataset:
    def download(self, download_path: Path = Path("data/raw/utdataset")):
        """Download and extract dataset from source.

        Parameters
        ----------
        - download_path (Path) :
            The path where to download the data.
        """

        zip_file_urls = self.load_download_urls()

        for zip_file_url in zip_file_urls:
            response = requests.get(zip_file_url, timeout=1000)
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_response:
                zip_response.extractall(download_path)

        self.move_videos(download_path)
        shutil.rmtree(download_path / "segmented_set1")
        shutil.rmtree(download_path / "segmented_set2")

        self.create_labels(download_path)

    def load_download_urls(self,
                           urls_path: Union[Path, None] = None
                           ) -> List[str]:
        """Return the urls where to download the dataset from.

        Parameters
        ----------
        - download_path (Path) :
            The path where the data have been downloaded.
        """

        if urls_path is None:
            urls_path = Path(__file__).parent / "download_urls.yaml"

        with urls_path.open('r', encoding='utf8') as urls_file:
            urls = yaml.safe_load(urls_file)

        return urls["UTDataset"]

    def move_videos(self, download_path: Path):
        """The archive contains two subfolders and this function
        move every video from the subfolders to the parent folder.

        Parameters
        ----------
        - download_path (Path) :
            The path where the data have been downloaded.
        """

        for video_path in download_path.glob("*/*.avi"):
            video_path.rename(download_path / video_path.name)

    ACTION_INDEXES = ["hand shaking", "hugging", "kicking",
                      "pointing", "punching", "pushing"]

    def create_labels(self, download_path: Path):
        """generate a label file from the name of the video
        which contains: a sample number, a sequence number, and
        the action number.

        Parameters
        ----------
        - download_path (Path) :
            The path where the data have been downloaded.
        """

        for video_path in download_path.glob("*.avi"):
            _, _, action = video_path.stem.split("_")
            corresponding_action = self.ACTION_INDEXES[int(action)]

            label_path = download_path / (video_path.stem + ".txt")
            with label_path.open("w", encoding="utf8") as label_file:
                label_file.write(corresponding_action)
