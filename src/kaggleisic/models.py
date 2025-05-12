import torch
import torch.nn as nn
from torchvision import models


class ResNet50Multimodal(nn.Module):
    def __init__(self, metadata_input_dim: int, dropout: float = 0.3):
        super().__init__()

        # Load pre-trained ResNet-50
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # remove original classifier (2048-dim output)

        # Metadata processing branch
        self.metadata_net = nn.Sequential(
            nn.Linear(metadata_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Binary classification
        )

    def forward(self, image, metadata):
        image_feats = self.resnet(image)  # (batch, 2048)
        metadata_feats = self.metadata_net(metadata)  # (batch, 64)
        combined = torch.cat([image_feats, metadata_feats], dim=1)
        return self.classifier(combined)


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
