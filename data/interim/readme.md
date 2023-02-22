This directory holds all the intermediate data e.g. videos with different frame rates into frames.

# Structure

The interim folder is organised as follows:
```
. (interim)
│
├── Kicking
│   │   # videos with individual actions
│   ├── sequence_1 5_fps 11_frames
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   ├── sequence_1 10_fps 22_frames
│   │   └── ...
│   ├── sequence_2 5_fps 14_frames
│   │   └── ...
│   └── ...
│
├── Neutral
│   │   # videos with individual neutral actions
│   └── ...
│
├── Continuous
│   │   # long videos with multiple actions
│   ├── sequence_1 5_fps 1245_frames
│   │   ├── labels.json
│   │   │   # labels in the form of a list
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   ├── sequence_1 10_fps 2490_frames
│   │   └── ...
│   └── ...
│
└── ...
```
