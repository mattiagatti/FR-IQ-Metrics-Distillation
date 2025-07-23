import timm
import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='avg')

        # Use self.backbone.num_features unless it's MobileNetV3
        if 'mobilenetv3' not in model_name.lower():
            num_features = self.backbone.num_features
        else:
            num_features = 1280

        self.head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)