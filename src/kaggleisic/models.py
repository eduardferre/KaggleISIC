import torch
import torch.nn as nn
from torchvision import models


class MLP_MetadataOnly(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),  # Batch normalization for the hidden layer
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),  # Batch normalization for the second hidden layer
            nn.Linear(64, 1),  # Single output unit for binary classification
        )

    def forward(self, metadata):
        return self.mlp(metadata)


class MLP_ImageOnly(nn.Module):
    def __init__(self, image_shape=(3, 224, 224)):
        super().__init__()
        flattened_size = image_shape[0] * image_shape[1] * image_shape[2]
        self.mlp = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, image):
        x = image.view(image.size(0), -1)
        return self.mlp(x)


class MLP_Multimodal(nn.Module):
    def __init__(self, num_metadata_features, image_shape=(3, 224, 224)):
        super().__init__()
        flattened_image_size = image_shape[0] * image_shape[1] * image_shape[2]
        input_size = flattened_image_size + num_metadata_features

        self.mlp = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, image, metadata):
        image_flat = image.view(image.size(0), -1)
        combined = torch.cat((image_flat, metadata), dim=1)
        return self.mlp(combined)


class ResNet50_Simple_ImageOnly(nn.Module):
    def __init__(self, out_features=1):
        super(ResNet50_Simple_ImageOnly, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, out_features)

    def forward(self, x):
        return self.resnet(x)


class ResNet50_Custom_ImageOnly(nn.Module):
    def __init__(self, out_features=1, dropout_rate=0.5):
        super(ResNet50_Custom_ImageOnly, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, out_features),
        )

    def forward(self, x):
        return self.resnet(x)
