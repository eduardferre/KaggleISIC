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
