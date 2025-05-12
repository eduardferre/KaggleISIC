import uuid
from urllib.parse import urlparse

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from kaggleisic import config

ORIGINAL_HDF5_PATH = urlparse(str(config.RAW_DATA_DIR / "train-image.hdf5")).path
OUTPUT_HDF5_PATH = urlparse(
    str(config.PROCESSED_DATA_DIR / "train-image-augmented.hdf5")
).path

AUGMENTATIONS = T.Compose(
    [
        T.Resize((224, 224)),  # Ensure consistent size
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=45, interpolation=TF.InterpolationMode.BILINEAR),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        T.RandomPerspective(
            distortion_scale=0.2,
            p=0.2,
            interpolation=TF.InterpolationMode.BILINEAR,
        ),
    ]
)


load_dotenv()


def generate_augmented_cancer_data(
    train_meta_df: pd.DataFrame,
    augmentations=AUGMENTATIONS,
    n_augments=3,
):
    cancer_df = train_meta_df[train_meta_df["target"] == 1].copy()
    new_rows = []

    with h5py.File(OUTPUT_HDF5_PATH, "a") as new_hf:
        with h5py.File(ORIGINAL_HDF5_PATH, "r") as hf:  # Read original images once
            for idx, row in tqdm(cancer_df.iterrows(), total=len(cancer_df)):
                isic_id = row["isic_id"]

                # Load original image
                image_bytes = hf[isic_id][()]
                image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)

                for _ in range(n_augments):
                    aug_image = augmentations(image_pil)

                    # Convert back to NumPy + encode
                    aug_np = np.array(aug_image)
                    aug_bgr = cv2.cvtColor(aug_np, cv2.COLOR_RGB2BGR)
                    success, encoded_aug = cv2.imencode(".jpg", aug_bgr)
                    encoded_bytes = encoded_aug.tobytes()

                    new_isic_id = f"ISIC_AUG_{uuid.uuid4().hex[:7]}"
                    new_hf.create_dataset(
                        new_isic_id, data=np.frombuffer(encoded_bytes, dtype="uint8")
                    )

                    # Create new metadata row
                    new_row = row.copy()
                    new_row["isic_id"] = new_isic_id
                    new_rows.append(new_row)

    print(f"✅ Augmented {len(new_rows)} samples.")

    return pd.concat([train_meta_df, pd.DataFrame(new_rows)], ignore_index=True)
