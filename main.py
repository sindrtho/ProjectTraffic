import os
import torch
import torchvision

from torch.utils import data
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from coco_dataset import CocoDataset
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import save_image
from torchvision.transforms.functional import convert_image_dtype
from movie_utils import export_video_from_frames
from utils_our import delete_everything_in_directory
from engine import evaluate


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform():
    custom_transforms = [torchvision.transforms.ToTensor()]
    return torchvision.transforms.Compose(custom_transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    # Config
    num_epochs = 10
    num_classes = 9
    train_batch_size = 1

    # Images and annotations for training
    image_folders = [
        'data/ambient/000000_ambient',
        'data/ambient/000001_ambient',
        'data/ambient/000002_ambient',
        'data/ambient/000003_ambient',
        'data/ambient/000004_ambient',
        # 'data/ambient/000005_ambient',
        'data/ambient/000006_ambient',
        'data/ambient/000007_ambient',
        'data/ambient/000008_ambient',
        'data/ambient/000009_ambient',
        'data/ambient/000010_ambient',
        'data/ambient/000011_ambient',
        'data/ambient/000012_ambient',
        # 'data/ambient/000013_ambient',
        # 'data/ambient/000014_ambient',
        'data/ambient/000015_ambient',
        'data/ambient/000016_ambient',
        'data/ambient/000017_ambient',
        'data/ambient/000018_ambient',
    ]
    annotation_files = [
        'data/annotations/000000_coco.json',
        'data/annotations/000001_coco.json',
        'data/annotations/000002_coco.json',
        'data/annotations/000003_coco.json',
        'data/annotations/000004_coco.json',
        # 'data/annotations/000005_coco.json',
        'data/annotations/000006_coco.json',
        'data/annotations/000007_coco.json',
        'data/annotations/000008_coco.json',
        'data/annotations/000009_coco.json',
        'data/annotations/000010_coco.json',
        'data/annotations/000011_coco.json',
        'data/annotations/000012_coco.json',
        # 'data/annotations/000013_coco.json',
        # 'data/annotations/000014_coco.json',
        'data/annotations/000015_coco.json',
        'data/annotations/000016_coco.json',
        'data/annotations/000017_coco.json',
        'data/annotations/000018_coco.json',
    ]

    # Images and annotations for evaluation
    evaluation_images = ['data/ambient/000013_ambient', 'data/ambient/000014_ambient']
    evaluation_annotations = ['data/annotations/000013_coco.json', 'data/annotations/000014_coco.json']

    # Dataset for training
    datasets = [CocoDataset(root=X, annotation=Y, transforms=get_transform()) for X, Y in
                zip(image_folders, annotation_files)]

    # Dataset for evaluation
    eval_datasets = [CocoDataset(root=X, annotation=Y, transforms=get_transform()) for X, Y in
                     zip(evaluation_images, evaluation_annotations)]

    # List of data loaders for training
    data_loaders = [torch.utils.data.DataLoader(dataset,
                                                batch_size=train_batch_size,
                                                shuffle=True,
                                                num_workers=4,
                                                collate_fn=collate_fn) for dataset in datasets]

    # List of data loaders for evaluation
    eval_data_loader = [torch.utils.data.DataLoader(eval_dataset,
                                                    batch_size=train_batch_size,
                                                    shuffle=False,
                                                    num_workers=4,
                                                    collate_fn=collate_fn) for eval_dataset in eval_datasets]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.5, weight_decay=0.0005)
    len_dataloaders = [len(data_loader) for data_loader in data_loaders]

    for epoch in range(num_epochs):
        model.train()
        for loader_index, loader in enumerate(data_loaders, 0):
            for iteration, (images, targets) in enumerate(loader, 0):
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                predictions = model(images, targets)
                losses = sum(loss for loss in predictions.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                print(
                    f'\033[92mEpoch: {epoch + 1}/{num_epochs}\033[00m Iteration: {sum(len_dataloaders[:loader_index]) + iteration}/{sum(len_dataloaders)}, Loss: {losses}')

    # Making directory for predictions and deleting everything from last run
    os.makedirs('predictions', exist_ok=True)
    delete_everything_in_directory('./predictions')

    model.eval()
    for eval_data in eval_data_loader:
        evaluate(model, eval_data, device=device)

    i = 0
    torch.set_printoptions(profile="full")
    for loader_index, loader in enumerate(eval_data_loader, 0):
        directory_images = './predictions/' + str(i) + '/'
        os.makedirs(directory_images, exist_ok=True)
        image_id = 0
        for iteration, (images, targets) in enumerate(loader, 0):
            image = images[0]
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            predictions = model(images)
            # print(predictions)
            boxes = predictions[0]['boxes']
            scores = predictions[0]['scores']
            labels = predictions[0]['labels'][scores > 0.7].tolist()
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
