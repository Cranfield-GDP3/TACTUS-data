from typing import Union, List
from pathlib import Path
import yaml

DEFAULT_URL_PATH = Path(__file__).parent.parent / "dataset_urls.yaml"


def read_dataset_urls(key: Union[str, None] = None,
                      urls_path: Path = DEFAULT_URL_PATH,
                      ) -> List[str]:
    """Return the URLs where to download the dataset from
    that correspond to the key.

    Parameters
    ----------
    - key (str):
        The URLs key (e.g. 'UTDataset')
    - urls_path (Path):
        Path to the YAML file containing the URLs.
    """

    with urls_path.open('r', encoding='utf8') as urls_file:
        urls = yaml.safe_load(urls_file)

    if key is None:
        return urls

    return urls[key]
