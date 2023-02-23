def interim_name_convention(source_name: str, uid: str, fps: int) -> str:
    """
    generate the name for the folder created for a video that is to be
    transformed into frames.

    Parameters
    ----------
    source_name : str
        dataset name
    uid : str
        unique id for the video in the dataset
    fps : int
        target fps

    Returns
    -------
    str
        formatted as the following: 'ut_interaction 1_11 5_fps'
    """

    return f"{source_name} {uid} {fps}_fps"
