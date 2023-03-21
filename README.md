<br />
<div align="center">
  <h1>TACTUS - data pipeline</h1>
  <h3>Threatening activities classification toward users' security</h3>
</div>

# Useful ressources

- [Write a better commit message](https://gist.github.com/MarcBresson/dd57a17f2ae60b6cb8688ee64cd7671d)
- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)

# Installation

## Hardware accelerated Torch

You can install torch and torchvision **with cuda** via pip. Get the pip install command from [the official website](https://pytorch.org/get-started/locally/).

## tactus-data

You can create an editable install of this library ([see more](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs)) with the following command:

```bash
python -m pip install -e .
```

The library will be added to your Python's import path and  you will be able to import this package from anywhere using

```python
import tactus_data
```

# Data sources

|name    |description     |url    |handled in the data pipeline|
|--------|----------------|-------|----------------------------|
|UT-Interaction Dataset​|This dataset contains videos of continuous executions of 6 classes of human-human interactions: shake-hands, point, hug, push, kick and punch.|https://paperswithcode.com/dataset/ut-interaction|✅|
|Surveillance camera fights dataset​|This dataset is benefited for developing a fight detection system which is aimed to use in surveillance cameras in public areas such as streets, underground stations and more.|https://github.com/seymanurakti/fight-detection-surv-dataset|🚫|
|UCF-Crime​|It consists of 1900 long and untrimmed real-world surveillance videos, with 13 realistic anomalies including Abuse, Arrest, Arson, Assault, Road Accident, Burglary, Explosion, Fighting, Robbery, Shooting, Stealing, Shoplifting, and Vandalism. These anomalies are selected because they have a significant impact on public safety.​|https://paperswithcode.com/dataset/ucf-crime|🚫|
|Weapon detection dataset​|This dataset contains various categories of knives, guns, and other weapons with their annotations​|https://github.com/ari-dasci/OD-WeaponDetection|🚫|
|Anomaly Detection Dataset​|This dataset contains training and test videos for abuse detection​|https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0|🚫|
|Hockey Fight Detection Dataset​|It’s a video database containing 1000 sequences divided in two groups: fights and non-fights​|https://paperswithcode.com/dataset/hockey-fight-detection-dataset|🚫|
|Real life violence situation dataset|This dataset contains about 1000 videos of real street fights as well as non-fights.​|https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset|[🚫](https://storage.googleapis.com/kaggle-data-sets/176381/397693/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230125%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230125T183716Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=84ce15191672067fd58e195f23475359a99ffb0bb89520410fb6ef52b9fa1b83d0c8c2c4b68af300de20b2c4905d5971318dfafef2e09fa705704d2b5ea1c5f3810a551a6f9cc24533e1db517865c463bc6d5f54a672cf0382156f5a553ee232c91558306f0ad2385c3ba8d9d0b34738005104183ae433e44642c5ad9a38f29cce61bfb7dcce4280570e4545aa159892c57d915360a5e84606a778a628424d925425fb3dec6cceec711e5885a48a255ae9a0edc78c0b55e40454081fc797fe3d0e3cffa60aa784c8589b6175a3cc13259402808107c9b05e0010d05dc41907e84f016a4473aaec5f71317fa309e528d5b1d4a9ae121cea5bc5b22fe69ca2c0f8)
