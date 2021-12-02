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
    train_data_dir = 'data/ambient/000000_ambient'
    train_coco = 'data/annotations/000000_coco.json'

    my_dataset = CocoDataset(root=train_data_dir, annotation=train_coco, transforms=get_transform())

    train_batch_size = 1
    data_loader = torch.utils.data.DataLoader(my_dataset,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 8
    num_epochs = 10

    model = get_model_instance_segmentation(num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    len_dataloader = len(data_loader)

    for epoch in range(num_epochs):
        model.train()

        i = 0
        for imgs, annotations in data_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')


if __name__ == '__main__':
    main()
