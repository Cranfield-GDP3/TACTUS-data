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
    "min_nbr_skeletons": 1,
    "max_nbr_skeletons": 2,
    "frames": [
        {
            "frame_id": "000.jpg",
            "skeletons": [
                {
                    "keypoints": [373,47,0.91,379,43,0.96,374,41,0.23,396,48,0.98,386,41,0.07,409,82,0.97,389,69,0.84,412,129,0.94,384,108,0.25,403,168,0.89,376,142,0.23,400,159,0.97,386,153,0.92,403,212,0.97,392,205,0.91,406,269,0.94,394,252,0.88],
                    "id_stupid": 2
                },
                {
                    "keypoints": [154,46,0.9,152,41,0.08,150,42,0.96,128,43,0.01,136,49,0.92,111,69,0.9,126,86,0.98,127,112,0.47,129,134,0.95,148,133,0.43,150,166,0.9,114,156,0.95,128,164,0.98,114,213,0.93,129,222,0.97,108,265,0.9,123,280,0.94],
                    "id_stupid": 1
                }
            ]
        },
        {
            "frame_id": "003.jpg",
            "skeletons": [
                {
                    "keypoints": [373,47,0.91,379,43,0.96,374,41,0.22,396,48,0.98,386,41,0.07,409,82,0.97,389,69,0.83,412,129,0.94,385,109,0.24,403,168,0.89,378,143,0.22,400,159,0.97,386,153,0.92,403,212,0.97,392,205,0.9,406,269,0.94,394,252,0.88],
                    "id_stupid": 2
                },
                {
                    "keypoints": [154,46,0.9,152,41,0.08,150,42,0.95,128,43,0.01,136,49,0.92,111,69,0.91,126,86,0.98,127,112,0.47,129,134,0.95,148,133,0.43,150,166,0.9,114,156,0.95,128,165,0.98,114,213,0.93,129,222,0.97,108,265,0.9,123,280,0.94],
                    "id_stupid": 1
                }
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
