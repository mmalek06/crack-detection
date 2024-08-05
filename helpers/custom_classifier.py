import torch
import torch.nn as nn


class CustomClassifier(nn.Module):
    def __init__(
            self,
            input_shape: tuple[int, int, int] = (3, 224, 224),
            conv_out_shapes: tuple[int, int] = (64, 128),
            linear_layers_features: int = 512
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, conv_out_shapes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(conv_out_shapes[0], conv_out_shapes[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
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

