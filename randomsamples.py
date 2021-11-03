import os
import random as r

"""
Script for picking out random demo images
"""

# Check wether on windows or linux operative system
system = os.name

# Number of desiered images
NUMBER_OF_IMAGES = 30

# Path to source images and destination folder
IMAGE_PATH = '.\\datasets\\RDD2020_filtered\\JPEGImages\\' if os.name == 'nt' else './datasets/RDD2020_filtered/JPEGImages/'
TARGET_DIRECTORY = '.\\demo\\rdd2020\\' if system == 'nt' else './demo/rdd2020/'

# Check if destination exists
if not os.path.exists(TARGET_DIRECTORY):
	print(f"Directory {TARGET_DIRECTORY} not found.\nCreating directory {os.path.abspath('.') + TARGET_DIRECTORY}")
	os.makedirs(TARGET_DIRECTORY)

images = []

# Find all available images
for path, folders, files in os.walk(IMAGE_PATH):
	for image in files:
		images.append(image)

# Pick NUMBER_OF_IMAGES random images to copy
to_copy = [images.pop(r.randint(0, len(images)-1)) for i in range(NUMBER_OF_IMAGES)]

del(images)

sources = [''.join([IMAGE_PATH, image]) for image in to_copy]
targets = [''.join([TARGET_DIRECTORY, image]) for image in to_copy]

# Delete old demo data from directory if exists
prev = os.listdir(TARGET_DIRECTORY)
if len(prev) != 0:
	print(f"Removing {len(prev)} old files from target directory")
	print("-"*43)
	for im in prev:
		print(f"Removing {im}")
		os.system(f"del /q {TARGET_DIRECTORY}\\{im}" if system == 'nt' else f"rm -rf {TARGET_DIRECTORY}/{im}")

print("\n\n")
print(f"Copying {NUMBER_OF_IMAGES} to target directory {TARGET_DIRECTORY}")
print("-"*(31+len(TARGET_DIRECTORY)))
for source, target in zip(sources, targets):
	print(f"{source} =====> {target}")
	os.system(f"copy {source} {target} >nul" if system == 'nt' else f"cp {source} {target}")