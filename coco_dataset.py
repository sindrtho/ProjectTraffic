import os
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        image_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=image_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(image_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path))
        num_objs = len(coco_annotation)

        boxes = []

        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((num_objs), dtype=torch.int64)

        image_id = torch.tensor([image_id])

        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        iscrowd = torch.zeros((num_objs), dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'area': areas, 'iscrowd': iscrowd}

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)
