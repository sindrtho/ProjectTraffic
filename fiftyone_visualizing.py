import fiftyone as fo

name = 'my-dataset'
dataset_dir = './data_fiftyone'
dataset_type = fo.types.COCODetectionDataset

dataset = fo.Dataset.from_dir(dataset_dir=dataset_dir, dataset_type=dataset_type, name=name)
session = fo.launch_app(dataset)
session.wait()
