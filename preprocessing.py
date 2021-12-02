import os
import pathlib
import re
import shutil
import zipfile
from tqdm import tqdm
import cv2
import moviepy.editor as mp


def get_files_with_regex(path, regex):
    result = []
    for root, _, files in os.walk(path):
        for file in files:
            found_file = re.search(regex, file)
            if found_file:
                file_path = os.path.join(root, file)
                result.append(file_path)
    return result


def preprocess_annotations(annotation_paths, annotations_path='./data/annotations'):
    for coco_path in tqdm(annotation_paths):
        os.makedirs(annotations_path, exist_ok=True)
        shutil.copy(coco_path, annotations_path)

        with zipfile.ZipFile(coco_path, 'r') as zip_ref:
            zip_ref.extractall(annotations_path)

        base = os.path.basename(coco_path)
        os.remove(os.path.join(annotations_path, base))

        for root, _, files in os.walk('./data/annotations/annotations'):
            for file in files:
                annotation_path_old = os.path.join(root, file)

                filename = '0000' + os.path.splitext(base)[0] + '.json'
                annotations_path_new = os.path.join(annotations_path, filename)

                shutil.move(annotation_path_old, annotations_path_new)
    os.rmdir('./data/annotations/annotations/')


def preprocess_videos(video_paths):
    for video_path in tqdm(video_paths, desc="Extracting video frames"):
        base = os.path.basename(video_path)
        lidar_file = os.path.splitext(base.split('Video')[1])
        lidar_type = os.path.splitext(base.split('_')[1])[0]
        lidar_path = os.path.join('./data/' + lidar_type, '0' + lidar_file[0])
        os.makedirs(lidar_path, exist_ok=True)

        with mp.VideoFileClip(video_path) as video:
            for frame_idx, frame in enumerate(video.iter_frames()):
                impath = pathlib.Path(lidar_path, "frame_%.6d.PNG" % frame_idx)
                cv2.imwrite(str(impath), frame[:, :, ::-1])


if __name__ == '__main__':
    raw_data_path = './data/LiDAR-videos'
    coco_files = get_files_with_regex(path=raw_data_path, regex='coco\.zip$')
    avi_files = get_files_with_regex(path=raw_data_path, regex='\.avi$')
    preprocess_annotations(coco_files)
    preprocess_videos(avi_files)
