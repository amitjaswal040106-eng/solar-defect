import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    """4 conv blocks trained from scratch (baseline)."""
    def __init__(self, num_classes=2):
        super().__init__()
        layers, in_ch = [], 3
        for out_ch in (16, 32, 64, 128):
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1),
                       nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False),
                       nn.MaxPool2d(2)]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                        nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def get_model(name):
    if name == "cnn":
        return SmallCNN()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m
    raise ValueError(name)


def cam_layer(model, name):
    return model.layer4[-1] if name == "resnet18" else model.features[-2]