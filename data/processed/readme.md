This directory holds all the processed data i.e. skeletons cleaned, augmented and ready to be trained on.

# Structure

The processed folder is organised as follows:
```
. (processed)
│
├── ut_interaction
│   ├── 0_1_4
│   │   ├── labels.json
│   │   └── 10fps
│   │   │  ├── alphapose_2d.json
│   │   │  ├── alphapose_2d data_augment 1.json
│   │   │  ├── alphapose_2d data_augment 2.json
│   │   │  ├── alphapose_2d data_augment 3.json
│   │   │  └── ...
│   │   └── 5fps
│   │      └── ...
│   └── 0_11_4
│       ├── labels.json
│       ├── 10fps
│       │   ├── alphapose_2d.json
│       │   └── ...
│       └── 5fps
│          └── ...
└── ...
```

# JSON dictionnary

## skeletons information

This structure doesn't come from alphapose nor any other pose extraction algorithm. We developed it for the purpose of this project. Other keys can be added in the dictionnaries (alongside frame_id, or keypoints for instance).

```json
{
    "resolution": [324,218],
    "frames": [
        {
            "frame_id":"003.jpg",
            "skeletons":[
                {"id":0, "keypoints": [0.1, 0.025, 0.45, "..."]},
                {"id":1, "keypoints": [0.78, 0.452, 0.123, "..."]},
                {"id":3, "keypoints": [0.45, 0.867, 0.56, "..."]}
            ]
        },
        {
            "frame_id":"006.jpg",
            "skeletons":[
                "..."
            ]
        },
        "..."
    ]
}
```

## label information

```json
{
    "resolution": [324,218],
    "classes": [
        {
            "classification": "kicking",
            "start_frame": 48,
            "end_frame": 86
        },
        {
            "classification": "punching",
            "start_frame": 231,
            "end_frame": 512
        },
        "..."
    ]
}
```
