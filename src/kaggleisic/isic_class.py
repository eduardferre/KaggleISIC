import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F_v
from torch.utils.data import Dataset


class ISIC_HDF5_Dataset(Dataset):
    """
    A PyTorch Dataset that loads images from an HDF5 file given a DataFrame of IDs.
    Applies image transforms.
    """

    def __init__(
        self, df: pd.DataFrame, hdf5_path: str, transform=None, is_labelled: bool = True
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'isic_id' and optionally 'target'.
            hdf5_path (str): Path to the HDF5 file containing images.
            transform (callable): Optional transforms to be applied on a sample.
            is_labelled (bool): Whether the dataset includes labels (for train/val).
        """
        self.df = df.reset_index(drop=True)
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.is_labelled = is_labelled

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        isic_id = row["isic_id"]

        # Load image from HDF5
        image_rgb = self._load_image_from_hdf5(isic_id)

        # Apply transforms (PIL-style transforms require converting np array to PIL, or we can do tensor transforms)
        if self.transform is not None:
            # Convert NumPy array (H x W x C) to a PIL Imag
            image_pil = F_v.to_pil_image(image_rgb)
            image = self.transform(image_pil)
        else:
            # By default, convert it to a PIL Image
            view_transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
            image_pil = F_v.to_pil_image(image_rgb)
            image = view_transform(image_pil)

        if self.is_labelled:
            label = row["target"]
            label = torch.tensor(label).float()
            return image, label, isic_id
        else:
            return image, isic_id

    def _load_image_from_hdf5(self, isic_id: str):
        """
        Loads and decodes an image from HDF5 by isic_id.
        Returns a NumPy array in RGB format (H x W x 3).
        """
        with h5py.File(self.hdf5_path, "r") as hf:
            encoded_bytes = hf[isic_id][()]  # uint8 array

        # Decode the image bytes with OpenCV (returns BGR)
        image_bgr = cv2.imdecode(encoded_bytes, cv2.IMREAD_COLOR)
        # Convert to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb


class ISIC_Multimodal_Dataset(Dataset):
    """
    A PyTorch Dataset that loads images from an HDF5 file and metadata from a DataFrame.
    Supports optional transforms and training/testing mode.
    """

    def __init__(self, df, hdf5_path: str, transform=None, is_labelled: bool = True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'isic_id', metadata features, and optionally 'target'.
            hdf5_path (str): Path to the HDF5 file containing images.
            transform (callable): Optional image transforms.
            is_labelled (bool): Whether the dataset includes labels (for training/validation).
        """
        self.df = df.reset_index(drop=True)
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.is_labelled = is_labelled

        # Identify metadata columns (exclude isic_id and target)
        self.metadata_cols = [
            col for col in self.df.columns if col not in ["isic_id", "target"]
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        isic_id = row["isic_id"]

        # --- Load and transform image ---
        image_rgb = self._load_image_from_hdf5(isic_id)

        if self.transform is not None:
            image_pil = F_v.to_pil_image(image_rgb)
            image = self.transform(image_pil)
        else:
            default_transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
            image_pil = F_v.to_pil_image(image_rgb)
            image = default_transform(image_pil)

        # --- Load metadata ---
        metadata = torch.tensor(row[self.metadata_cols].values.astype("float32"))

        if self.is_labelled:
            label = torch.tensor(row["target"]).float()
            return metadata, image, label
        else:
            return metadata, image, isic_id

    def _load_image_from_hdf5(self, isic_id: str):
        """
        Loads and decodes an image from HDF5 by isic_id.
        Returns a NumPy array in RGB format (H x W x 3).
        """
        with h5py.File(self.hdf5_path, "r") as hf:
            encoded_bytes = hf[isic_id][()]
        image_bgr = cv2.imdecode(encoded_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        return image_rgb
