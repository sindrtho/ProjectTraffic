import pathlib
import tqdm
import cv2
import argparse
import moviepy.editor as mp
import tempfile
import os


# Paths to video files, annotation files, and folder for temporary frame storage
DATA_PATH = './data/short_videos/'
ANNOTATION_PATH = './data/annotations/'
IMAGE_PATH = './data/images/'

VIDEOS = []
# Walk through folders and find all video files
for path, _, file in os.walk(DATA_PATH):
	for f in file:
		VIDEOS.append(path+f)

print(VIDEOS)
# Example of string formating to get the right video and annotation files.
VIDEO_FILE = 'Video%.5d_ambient.avi'
ANNOTATION_FILE = '%.2d_coco/annotations/instance_default.json'
for i in range(19):
	# print(VIDEO_FILE % i)
	# print(ANNOTATION_FILE % i)
	continue

v = VIDEOS[0]
with mp.VideoFileClip(v) as video:
	for frame_idx, frame in enumerate(
            tqdm.tqdm(video.iter_frames(), desc="Reading video frames")):
		impath = pathlib.Path(IMAGE_PATH, "frame_%.5d.png" % frame_idx)
		cv2.imwrite(str(impath), frame[:, :, ::-1])