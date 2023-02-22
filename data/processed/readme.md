This directory holds all the processed data i.e. skeletons cleaned, augmented and ready to be trained on.

# Structure

The processed folder is organised as follows:
```
. (processed)
│
├── Kicking
│   │   # skeleton sequences with individual actions
│   ├── sequence_1 5_fps 11_frames.json
│   ├── sequence_1 10_fps 22_frames.json
│   ├── sequence_2 5_fps 14_frames.json
│   └── ...
│
├── Neutral
│   │   # skeleton sequences with individual neutral actions
│   └── ...
│
├── Continuous
│   │   # long skeleton sequences with multiple actions
│   ├── sequence_1 5_fps 1245_frames.json
│   ├── sequence_1 10_fps 2490_frames.json
│   └── ...
│
└── ...
```

# JSON dictionnary
