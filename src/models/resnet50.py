import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.optim.lr_scheduler import StepLR

import kaggleisic.train_valid_eval as tve
from kaggleisic.dataloader import create_dataloaders
from kaggleisic.load_data import (
    load_hdf5_dataset,
    load_metadata_dataset,
    load_multimodal_dataset,
)
from kaggleisic.models import ResNet50Multimodal


def load_data(view_transform: T.Compose = None):
    if view_transform is None:
        view_transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    # Load the multimodal dataset
    train_multi_df, valid_multi_df, test_multi_df = load_multimodal_dataset(
        transform=view_transform
    )

    # Load processed metadata
    train_meta_df, valid_meta_df, test_meta_df = load_metadata_dataset()

    # Load HDF5 dataset
    train_hdf5_df, valid_hdf5_df, test_hdf5_df = load_hdf5_dataset(
        transform=view_transform
    )

    return (
        train_multi_df,
        valid_multi_df,
        test_multi_df,
        train_meta_df,
        valid_meta_df,
        test_meta_df,
        train_hdf5_df,
        valid_hdf5_df,
        test_hdf5_df,
    )


if __name__ == "__main__":
    # Load the data
    (
        train_multi_df,
        valid_multi_df,
        test_multi_df,
        train_meta_df,
        valid_meta_df,
        test_meta_df,
        train_hdf5_df,
        valid_hdf5_df,
        test_hdf5_df,
    ) = load_data()

    # Create dataloaders
    # train_loader, valid_loader, test_loader, full_loader = create_dataloaders(
    #     train_multi_df, valid_multi_df, test_multi_df, train_meta_df
    # )
    train_loader, valid_loader, test_loader, full_loader = create_dataloaders(
        train_hdf5_df, valid_hdf5_df, test_hdf5_df, train_meta_df
    )

    # Instantiate the model
    # model = ResNet50Multimodal(
    #     metadata_input_dim=len(train_meta_df.columns) - 1
    # )  # -1 for target column

    # # Train and validate the model
    # tve.train_valid_multimodal(model, train_loader, valid_loader)
