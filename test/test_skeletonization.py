from tactus_data import BodyAngles, Skeleton
from tactus_data import SkeletonRollingWindow


def test_rolling_window_velocity():
    input_keypoints = [
        [(0, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
        [(0, 1), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
        [(0, 1), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
        [(0, 1), (2, 2), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
        [(0, 1), (3, 4), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
    ]

    expected_velocity = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]

    rolling_window = SkeletonRollingWindow(5)

    for keypoints in input_keypoints:
        skeleton = Skeleton(keypoints=keypoints)
        rolling_window.add_skeleton(skeleton)

    velocity = [int(kpt) for velocity in rolling_window.velocities_rw for kpt in velocity]

    assert velocity == expected_velocity


def test_rolling_window_angles():
    input_keypoints = [
        [(0, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
        [(0, 1), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (2, 0), (0, 0), (0, 0), (0, 0), (0, 1), (0, 0)],
        [(0, 1), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0)],
        [(0, 1), (2, 2), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
        [(0, 1), (3, 4), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 2), (0, 0), (0, 0), (0, 0), (-3, 0), (0, 0)],
    ]

    expected_angle_LKnee_angle = [
        0,
        90,
        0,
        0,
        90
    ]

    rolling_window = SkeletonRollingWindow(5, angles_to_compute=[BodyAngles.LKnee.value])

    for keypoints in input_keypoints:
        skeleton = Skeleton(keypoints=keypoints)
        rolling_window.add_skeleton(skeleton)

    angles = [int(kpt) for angle in rolling_window.angles_rw for kpt in angle]

    assert angles == expected_angle_LKnee_angle
