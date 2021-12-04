import torch
import torchvision
from torch.utils import data
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from coco_dataset import CocoDataset


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
        # 'data/ambient/000006_ambient',
        # 'data/ambient/000007_ambient',
        # 'data/ambient/000008_ambient',
        # 'data/ambient/000009_ambient',
        # 'data/ambient/000010_ambient',
        # 'data/ambient/000011_ambient',
        # 'data/ambient/000012_ambient',
        # 'data/ambient/000013_ambient',
        # 'data/ambient/000014_ambient',
        # 'data/ambient/000015_ambient',
        # 'data/ambient/000016_ambient',
        # 'data/ambient/000017_ambient',
        ]
    annotation_files = [
        'data/annotations/000000_coco.json',
        # 'data/annotations/000001_coco.json',
        # 'data/annotations/000002_coco.json',
        # 'data/annotations/000003_coco.json',
        # 'data/annotations/000004_coco.json',
        # 'data/annotations/000006_coco.json',
        # 'data/annotations/000007_coco.json',
        # 'data/annotations/000008_coco.json',
        # 'data/annotations/000009_coco.json',
        # 'data/annotations/000010_coco.json',
        # 'data/annotations/000011_coco.json',
        # 'data/annotations/000012_coco.json',
        # 'data/annotations/000013_coco.json',
        # 'data/annotations/000014_coco.json',
        # 'data/annotations/000015_coco.json',
        # 'data/annotations/000016_coco.json',
        # 'data/annotations/000017_coco.json',
        ]

    evaluation_images, evaluation_annotations = 'data/ambient/000018_ambient', 'data/annotations/000018_coco.json'

    # Dataset for training
    datasets = [CocoDataset(root=X, annotation=Y, transforms=get_transform()) for X, Y in zip(image_folders, annotation_files)]

    # Dataset for evaluation
    eval_dataset = CocoDataset(root=evaluation_images, annotation=evaluation_annotations, transforms=get_transform())

    train_batch_size = 1
    # List of dataloaders for training
    data_loaders = [torch.utils.data.DataLoader(dataset,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              collate_fn=collate_fn) for dataset in datasets]

    # Dataloader for evaluation
    eval_data_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    num_classes = 9
    num_epochs = 10

    model = get_model_instance_segmentation(num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.5, weight_decay=0.0005)

    len_dataloaders = [len(data_loader) for data_loader in data_loaders]

    # for epoch in range(num_epochs):
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

            print(f'Iteration: {sum(len_dataloaders[:loader_index]) + iteration}/{sum(len_dataloaders)}, Loss: {losses}')



    predictions = None
    model.eval()
    # losses = sum(loss for loss in loss_dict.values())
    i = 0
    for imgs, annotations in eval_data_loader:
        i += 1
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs)
        predictions = loss_dict
        # predictions.append(loss_dict)
        # print(loss_dict)
        # losses = sum(loss for loss in loss_dict.values())
        #
        # optimizer.zero_grad()
        # losses.backward()
        # optimizer.step()

        # print(f'Iteration: {i}/{len(eval_data_loader)}, Loss: {losses}')

    print(predictions)

    

if __name__ == '__main__':
    main()
