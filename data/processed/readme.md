This directory holds all the processed data i.e. skeletons cleaned, augmented and ready to be trained on.

# Structure

The processed folder is organised as follows:
```
. (processed)
│
├── ut_interaction
│   ├── 0_1_4
│   │   ├── 0_1_4.labels.json
│   │   └── 10fps
│   │   │  ├── yolov7.json
│   │   │  ├── yolov7 data_augment 1.json
│   │   │  ├── yolov7 data_augment 2.json
│   │   │  ├── yolov7 data_augment 3.json
│   │   │  └── ...
│   │   └── 5fps
│   │      └── ...
│   └── 0_11_4
│       ├── 0_11_4.labels.json
│       ├── 10fps
│       │   ├── yolov7.json
│       │   └── ...
│       └── 5fps
│          └── ...
└── ...
```

# JSON dictionnary

## skeletons information

This structure was developed for the purpose of this project. Other keys can be added in the dictionnaries if needed (alongside frame_id, or keypoints for instance).

```json
{
    
    "min_nbr_skeletons": 1,
    "max_nbr_skeletons": 2,
    "resolution": [
        320,
        448
    ],
    "frames": [
        {
            "frame_id": "000.jpg",
            "skeletons": [
                {
                    "keypoints": [391, 45, 409, 82, 389, 69, 412, 129, 384, 108, 403, 168, 376, 142, 400, 159, 386, 153, 403, 212, 392, 205, 406, 269, 394, 252],
                    "id_stupid": 2
                },
                {
                    "keypoints": [132, 46, 111, 69, 126, 86, 127, 112, 129, 134, 148, 133, 150, 166, 114, 156, 128, 164, 114, 213, 129, 222, 108, 265, 123, 280],
                    "id_stupid": 1
                }
            ]
        },
        {
            "frame_id": "003.jpg",
            "skeletons": [
                {
                    "keypoints": [391, 45, 409, 82, 389, 69, 412, 129, 385, 109, 403, 168, 378, 143, 400, 159, 386, 153, 403, 212, 392, 205, 406, 269, 394, 252],
                    "id_stupid": 2
                },
                {
                    "keypoints": [132, 46, 111, 69, 126, 86, 127, 112, 129, 134, 148, 133, 150, 166, 114, 156, 128, 165, 114, 213, 129, 222, 108, 265, 123, 280],
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
    "resolution": [296, 400],
    "classes": [
        {
            "classification": "punching",
            "start_frame": 47,
            "end_frame": 91
        }
    ],
    "offender": [2]
}
```
