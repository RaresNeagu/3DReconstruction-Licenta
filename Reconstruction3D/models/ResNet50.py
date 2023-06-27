import torch
from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck

import config


class ResNet50(ResNet):

    def __init__(self, *args, **kwargs):
        self.features_dim = 0
        super().__init__(*args, **kwargs)
        if "resnet50" in config.PRETRAINED_ResNet50:
            self.load_state_dict(torch.load(config.PRETRAINED_ResNet50))
            print("Loaded pretrained ResNet50 weights")

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        res = super()._make_layer(block, planes, blocks, stride, dilate)
        self.features_dim += self.inplanes
        return res

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)

        return features


def resnet50():
    model = ResNet50(Bottleneck, [3, 4, 6, 3])
    state_dict = torch.load(config.PRETRAINED_ResNet50)
    model.load_state_dict(state_dict)
    print("Loaded pretrained ResNet50 weights")
    return model
