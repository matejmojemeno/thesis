import glob
import json


def video_number(video_path):
    """Find path indices from video path"""
    video_path = video_path.split("/")[-1]
    video_path = video_path[len("video") : -4]
    numbers = video_path.split("_")
    numbers = [int(number) for number in numbers]
    return numbers


def sort_video_paths(video_paths):
    """Sort video paths so they always remain in the same order."""
    return sorted(video_paths)


def get_video_paths():
    """Get all video paths in sorted order."""
    video_paths = glob.glob("data/videos/*.mp4")
    return sort_video_paths(video_paths)


def get_test_video_paths():
    """Get all test video paths in sorted order."""
    video_paths = glob.glob("data/test_videos/*.mp4")
    return sort_video_paths(video_paths)


def get_video_paths_arrays():
    with open("data/path_split.json", "r") as f:
        path_split = json.load(f)

    video_paths = get_video_paths()
    paths = []

    for video_path in video_paths:
        video, sperm, split = video_number(video_path)
        paths.append(path_split[video - 1][sperm][split])

    return paths


def get_test_video_paths_arrays():
    with open("data/path_split.json", "r") as f:
        path_split = json.load(f)

    video_paths = get_test_video_paths()
    paths = []

    for video_path in video_paths:
        video, sperm, split = video_number(video_path)
        paths.append(path_split[video - 1][sperm][split])

    return paths
