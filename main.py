import glob
import os
import shutil

import cv2
import torch
import torchvision
from torch.utils import data
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from coco_dataset import CocoDataset
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import save_image
from torchvision.transforms.functional import convert_image_dtype
from movie_utils import export_video_from_frames
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    # Images and annotations for training
    image_folders = [
        'data/ambient/000000_ambient',
        # 'data/ambient/000001_ambient',
        # 'data/ambient/000002_ambient',
        # 'data/ambient/000003_ambient',
        # 'data/ambient/000004_ambient',
        # # 'data/ambient/000005_ambient',
        # 'data/ambient/000006_ambient',
        # 'data/ambient/000007_ambient',
        # 'data/ambient/000008_ambient',
        # 'data/ambient/000009_ambient',
        # 'data/ambient/000010_ambient',
        # 'data/ambient/000011_ambient',
        # 'data/ambient/000012_ambient',
        # # 'data/ambient/000013_ambient',
        # # 'data/ambient/000014_ambient',
        # 'data/ambient/000015_ambient',
        # 'data/ambient/000016_ambient',
        # 'data/ambient/000017_ambient',
        # 'data/ambient/000018_ambient',
    ]
    annotation_files = [
        'data/annotations/000000_coco.json',
        # 'data/annotations/000001_coco.json',
        # 'data/annotations/000002_coco.json',
        # 'data/annotations/000003_coco.json',
        # 'data/annotations/000004_coco.json',
        # # 'data/annotations/000005_coco.json',
        # 'data/annotations/000006_coco.json',
        # 'data/annotations/000007_coco.json',
        # 'data/annotations/000008_coco.json',
        # 'data/annotations/000009_coco.json',
        # 'data/annotations/000010_coco.json',
        # 'data/annotations/000011_coco.json',
        # 'data/annotations/000012_coco.json',
        # # 'data/annotations/000013_coco.json',
        # # 'data/annotations/000014_coco.json',
        # 'data/annotations/000015_coco.json',
        # 'data/annotations/000016_coco.json',
        # 'data/annotations/000017_coco.json',
        # 'data/annotations/000018_coco.json',
    ]

    evaluation_images = ['data/ambient/000013_ambient', 'data/ambient/000014_ambient']
    evaluation_annotations = ['data/annotations/000013_coco.json', 'data/annotations/000014_coco.json']



    # Dataset for training
    datasets = [CocoDataset(root=X, annotation=Y, transforms=get_transform()) for X, Y in
                zip(image_folders, annotation_files)]

    # Dataset for evaluation
    eval_datasets = [CocoDataset(root=X, annotation=Y, transforms=get_transform()) for X, Y in
                     zip(evaluation_images, evaluation_annotations)]

    train_batch_size = 1
    # List of dataloaders for training
    data_loaders = [torch.utils.data.DataLoader(dataset,
                                                batch_size=train_batch_size,
                                                shuffle=True,
                                                num_workers=4,
                                                collate_fn=collate_fn) for dataset in datasets]

    # Dataloader for evaluation
    eval_data_loader = [torch.utils.data.DataLoader(eval_dataset,
                                                    batch_size=train_batch_size,
                                                    shuffle=False,
                                                    num_workers=4,
                                                    collate_fn=collate_fn) for eval_dataset in eval_datasets]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    num_classes = 9
    num_epochs = 1

    model = get_model_instance_segmentation(num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.5, weight_decay=0.0005)

    len_dataloaders = [len(data_loader) for data_loader in data_loaders]

    for epoch in range(num_epochs):
        print('\033[92mEpoch: ' + str(epoch) + '\033[00m')
        model.train()
        for loader_index, loader in enumerate(data_loaders, 0):
            for iteration, (images, targets) in enumerate(loader, 0):
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                # print(f'Iteration: {sum(len_dataloaders[:loader_index]) + iteration}/{sum(len_dataloaders)}, Loss: {losses}')

    os.makedirs('predictions', exist_ok=True)
    print('Slett')
    # files = glob.glob('./predictions/0/*')
    # for f in files:
    #     os.remove(f)
    #
    # files2 = glob.glob('./predictions/100/*')
    # for f in files2:
    #     os.remove(f)
    folder = './predictions/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    model.eval()
    i = 0
    for loader_index, loader in enumerate(eval_data_loader, 0):
        directory_images = './predictions/' + str(i) + '/'
        os.makedirs(directory_images, exist_ok=True)
        image_id = 0
        for iteration, (images, targets) in enumerate(loader, 0):
            image = images[0]
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images)
            boxes = loss_dict[0]['boxes']
            scores = loss_dict[0]['scores']
            labels = loss_dict[0]['labels'][scores > 0.7].tolist()
            colors = []

            for j in range(len(labels)):
                if labels[j] == 1:
                    colors.append('blue')
                if labels[j] == 2:
                    colors.append('yellow')
                if labels[j] == 3:
                    colors.append('orange')
                if labels[j] == 4:
                    colors.append('pink')
                if labels[j] == 5:
                    colors.append('red')
                if labels[j] == 6:
                    colors.append('green')
                if labels[j] == 7:
                    colors.append('purple')
                if labels[j] == 8:
                    colors.append('brown')

            image = convert_image_dtype(image=image, dtype=torch.uint8)
            result = draw_bounding_boxes(image=image, boxes=boxes[scores > 0.7], colors=colors)
            result = convert_image_dtype(image=result, dtype=torch.float)
            save_image(result, directory_images + '/img' + str(image_id) + '.png')
            image_id += 1

        i += 1

    export_video_from_frames('./predictions/0', filename='./predictions/0.mp4', fps=30)
    export_video_from_frames('./predictions/1', filename='./predictions/1.mp4', fps=30)


if __name__ == '__main__':
    main()
