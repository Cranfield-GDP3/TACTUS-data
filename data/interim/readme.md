This directory holds all the intermediate data e.g. videos with different frame rates into frames.

# Structure

The interim folder is organised as follows:
```
. (interim)
│
├── kicking
│   │   # videos with individual actions
│   ├── ut_interaction 01_11 5_fps
│   │   ├── 01.jpg
│   │   ├── 06.jpg
│   │   ├── ...
│   │   └── 56.jpg
│   ├── ut_interaction 01_11 10_fps
│   │   └── ...
│   ├── ut_interaction 02_11 5_fps
│   │   └── ...
│   └── ...
│
├── neutral
│   │   # videos with individual neutral actions
│   └── ...
│
├── continuous
│   │   # long videos with multiple actions
│   ├── hockey_violence 001 5_fps
│   │   ├── labels.json
│   │   │   # labels in the form of a list
│   │   ├── 0001.jpg
│   │   ├── 0006.jpg
│   │   └── ...
│   ├── hockey_violence 001 10_fps
│   │   └── ...
│   └── ...
│
└── ...
```
