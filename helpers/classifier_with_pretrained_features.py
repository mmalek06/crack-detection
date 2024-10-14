import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models import ResNeXt50_32X4D_Weights


class Resnext50BasedClassifier(nn.Module):
    def __init__(
            self,
            input_shape: tuple[int, int, int] = (3, 224, 224),
            linear_layers_features: int = 512
    ):
        super().__init__()

        self.feature_extractor = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._get_feature_size(input_shape), linear_layers_features),
            nn.ReLU(inplace=True),
            nn.Linear(linear_layers_features, linear_layers_features),
            nn.ReLU(inplace=True),
            nn.Linear(linear_layers_features, 1)
        )

    def _get_feature_size(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            features = self.feature_extractor(dummy_input)

            return features.numel()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)

        return x


class Resnext50BasedClassifierForProposals(nn.Module):
    def __init__(
            self,
            input_shape: tuple[int, int, int] = (3, 224, 224),
            linear_layers_features: int = 512
    ):
        super().__init__()

        self.feature_extractor = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._get_feature_size(input_shape), linear_layers_features),
            nn.ReLU(inplace=True),
            nn.Linear(linear_layers_features, linear_layers_features),
            nn.ReLU(inplace=True),
            nn.Linear(linear_layers_features, 1)
        )

    def _get_feature_size(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            features = self.feature_extractor(dummy_input)

            return features.numel()

    def forward(self, x):
        features = self.feature_extractor(x)
        class_scores = self.classifier(features)

        return class_scores
