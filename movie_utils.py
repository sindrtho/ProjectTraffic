import os
import re
import moviepy.editor as mpy


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


def export_video_from_frames(path, filename, fps=30):
    frames = preprocess_frames(path)
    clips = mpy.ImageSequenceClip(frames, load_images=True, fps=fps)
    clips.write_videofile(filename, fps=30)


def get_files_in_directory(path):
    result = []
    for root, _, files in os.walk(path):
        for file in files:
            result.append(file)
    return result


def preprocess_frames(path):
    files = get_files_in_directory(path)
    files.sort(key=natural_keys)
    frames = []
    for file in files:
        frames.append(os.path.join(path, file))
    return frames


