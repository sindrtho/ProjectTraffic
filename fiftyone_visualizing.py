import fiftyone as fo

'''
    Use this file to visualize the dataset.
    Source: https://towardsdatascience.com/how-to-work-with-object-detection-datasets-in-coco-format-9bf4fb5848a4
'''

name = 'my-dataset'
dataset_dir = './data_fiftyone'
dataset_type = fo.types.COCODetectionDataset

dataset = fo.Dataset.from_dir(dataset_dir=dataset_dir, dataset_type=dataset_type, name=name)
session = fo.launch_app(dataset)
session.wait()
