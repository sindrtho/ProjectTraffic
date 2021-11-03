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

# Walk through folders and find all video files
for path, _, file in os.walk(DATA_PATH):
	for f in file:
		print(path+f)

# Example of string formating to get the right video and annotation files.
VIDEO_FILE = 'Video%.5d_ambient.avi'
ANNOTATION_FILE = '%.2d_coco/annotations/instance_default.json'
for i in range(19):
	print(VIDEO_FILE % i)
	print(ANNOTATION_FILE % i)