import torch
from torch import nn
import torchvision.models as models

class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        self.print_counter = 1
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS
        model = models.resnet18(pretrained=True)

        self.preprocessor = torch.nn.ModuleList([
            model.conv1,
            model.bn1, 
            model.relu,
            model.maxpool,
            model.layer1
        ])

        self.layers = torch.nn.ModuleList([
            model.layer2,
            model.layer3,
            model.layer4,
#            torch.nn.Sequential(
#                nn.Conv2d(in_channels=image_channels, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
#                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(in_channels=256, out_channels=output_channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
#                nn.BatchNorm2d(output_channels[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#            ),torch.nn.Sequential(
#                nn.Conv2d(in_channels=output_channels[1], out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
#                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                nn.ReLU(inplace=True),
#               nn.Conv2d(in_channels=256, out_channels=output_channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
#               nn.BatchNorm2d(output_channels[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#            ),
#            torch.nn.Sequential(
#                nn.Conv2d(in_channels=output_channels[1], out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
#                nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(in_channels=1024, out_channels=output_channels[2], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
#                nn.BatchNorm2d(output_channels[2], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#            ),
            torch.nn.Sequential(
                nn.Conv2d(in_channels=output_channels[2], out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=512, out_channels=output_channels[3], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(output_channels[3], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ),
            torch.nn.Sequential(
                nn.Conv2d(in_channels=output_channels[3], out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=256, out_channels=output_channels[4], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(output_channels[4], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ),
            torch.nn.Sequential(
                nn.Conv2d(in_channels=output_channels[4], out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=output_channels[5], kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
                nn.BatchNorm2d(output_channels[5], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            )
        ])

        """self.layers = torch.nn.ModuleList([
                torch.nn.Sequential(    # First layer
                    nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=64, out_channels=output_channels[0], kernel_size=3, stride=2, padding=1)
                ),
                torch.nn.Sequential(    # Second layer
                    nn.ReLU(),
                    nn.Conv2d(in_channels=output_channels[1-1], out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=128, out_channels=output_channels[2-1], kernel_size=3, stride=2, padding=1),
                ),
                torch.nn.Sequential(    # Third layer
                    nn.ReLU(),
                    nn.Conv2d(in_channels=output_channels[2-1], out_channels=256, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=256, out_channels=output_channels[3-1], kernel_size=3, stride=2, padding=1),
                ),
                torch.nn.Sequential(    # Fourth layer
                    nn.ReLU(),
                    nn.Conv2d(in_channels=output_channels[3-1], out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=128, out_channels=output_channels[4-1], kernel_size=3, stride=2, padding=1),
                ),
                torch.nn.Sequential(    # Fifth layer
                    nn.ReLU(),
                    nn.Conv2d(in_channels=output_channels[4-1], out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=128, out_channels=output_channels[5-1], kernel_size=3, stride=2, padding=1),
                ),
                torch.nn.Sequential(    # Sixth layer
                    nn.ReLU(),
                    nn.Conv2d(in_channels=output_channels[5-1], out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=128, out_channels=output_channels[6-1], kernel_size=3, stride=2, padding=1),
                )
            ])"""

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []

        res = x

        for preprocessor in self.preprocessor:
            res = preprocessor(res)

        for layer in self.layers:
            res = layer(res)
            out_features.append(res)

        if self.print_counter == 1:
            for r in out_features:
                print(r.shape)
            self.print_counter = 0

        channel_index = 0
        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[channel_index], h, w)
            channel_index += 1
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)