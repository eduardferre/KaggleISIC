import torch
import torch.nn as nn
import torch.nn.functional as F
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


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=7, stride=4, padding=3
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return x


# ConvNeXt block
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return input + x


# Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x_res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + x_res
        x_res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + x_res
        return x


class HybridConvNeXtViT(nn.Module):

    def __init__(self, num_classes=1):
        super().__init__()

        self.down1 = DownsampleBlock(3, 64)
        self.stage1 = nn.Sequential(*[ConvNeXtBlock(64) for _ in range(4)])

        self.down2 = DownsampleBlock(64, 128)
        self.stage2 = nn.Sequential(*[ConvNeXtBlock(128) for _ in range(8)])

        self.down3 = DownsampleBlock(128, 256)
        self.stage3 = nn.Sequential(
            *[TransformerBlock(256, num_heads=4) for _ in range(16)]
        )

        self.down4 = DownsampleBlock(256, 512)
        self.stage4 = nn.Sequential(
            *[TransformerBlock(512, num_heads=8) for _ in range(4)]
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.down1(x)
        x = self.stage1(x)

        x = self.down2(x)
        x = self.stage2(x)

        x = self.down3(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.stage3(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.down4(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.stage4(x)

        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.pool(x)
        x = x.view(B, -1)
        return self.fc(x)  # Output logits


class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64 * 28 * 28, latent_dim)
        self.fc_logvar = nn.Linear(64 * 28 * 28, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 64 * 28 * 28)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid(),  # Sigmoid ensures output is between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Ensure input images are between 0 and 1
        x = x / 255.0  # Normalize here if images are in range [0, 255]

        h1 = self.encoder(x)
        h1 = h1.view(h1.size(0), -1)
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        z = self.reparameterize(mu, logvar)
        h2 = self.decoder_fc(z)
        h2 = h2.view(-1, 64, 28, 28)
        x_reconstructed = self.decoder_conv(h2)

        # Upscale the output to (224, 224)
        x_reconstructed = F.interpolate(
            x_reconstructed, size=(224, 224), mode="bilinear", align_corners=False
        )

        return x_reconstructed, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # Binary Cross-Entropy
        BCE = nn.functional.binary_cross_entropy(
            recon_x, x.view(-1, 3, 224, 224), reduction="sum"
        )
        # Kullback-Leibler divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD
