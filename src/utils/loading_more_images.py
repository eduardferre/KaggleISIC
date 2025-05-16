import os
from urllib.parse import urlparse

import h5py
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PIL import Image

from kaggleisic import config

load_dotenv()

# === Paths ===
image_folder = urlparse(str(config.RAW_DATA_DIR / "ISIC-images")).path
metadata_file = os.path.join(image_folder, "metadata.csv")
test_metadata_file = config.RAW_DATA_DIR / "students-test-metadata.csv"
extra_metadata_file = config.RAW_DATA_DIR / "new-train-metadata.csv"
extra_image_file = urlparse(str(config.RAW_DATA_DIR / "train-image.hdf5")).path

output_hdf5_path = urlparse(str(config.INTERIM_DATA_DIR / "malign-image.hdf5")).path
output_meta_path = urlparse(str(config.INTERIM_DATA_DIR / "malign-metadata.csv")).path

resize_shape = (224, 224)

# === Load test isic_ids to exclude ===
print("Loading test metadata...")
test_metadata = pd.read_csv(test_metadata_file)
test_isic_ids = set(test_metadata["isic_id"].astype(str))
print(f"Found {len(test_isic_ids)} test images to exclude")

# === Load and filter original metadata (exclude test set) ===
metadata = pd.read_csv(metadata_file)[["isic_id"]]
metadata["target"] = 1  # Assume all are malignant
filtered_metadata = metadata[~metadata["isic_id"].isin(test_isic_ids)]

# === Process images from original folder ===
valid_image_data = []
valid_metadata = []

for _, row in filtered_metadata.iterrows():
    isic_id = row["isic_id"]
    image_filename = f"{isic_id}.jpg"
    image_path = os.path.join(image_folder, image_filename)

    if os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        img = img.resize(resize_shape)
        valid_image_data.append(np.asarray(img))
        valid_metadata.append(row)
    else:
        print(f"⚠️ Missing image: {image_filename}")

# === Record used isic_ids to avoid duplicates ===
used_isic_ids = set([row["isic_id"] for row in valid_metadata])

# === Load extra metadata and filter: benign + not in test + not already used ===
extra_metadata = pd.read_csv(extra_metadata_file)
extra_metadata = extra_metadata[
    (extra_metadata["target"] == 0)
    & (~extra_metadata["isic_id"].isin(test_isic_ids))
    & (~extra_metadata["isic_id"].isin(used_isic_ids))
].reset_index(drop=True)

# === Load matching images from extra HDF5 ===
with h5py.File(extra_image_file, "r") as hf:
    extra_images_all = hf["images"][:]

# Assumes image ordering matches extra_metadata
extra_images = []
for i, row in extra_metadata.iterrows():
    if i >= len(extra_images_all):
        print(f"⚠️ Index {i} out of bounds in train-image.hdf5, skipping")
        continue
    extra_images.append(extra_images_all[i])
    valid_metadata.append(row)

# === Combine image data ===
all_images = np.concatenate(
    [np.stack(valid_image_data), np.stack(extra_images)], axis=0
)
all_metadata = pd.DataFrame(valid_metadata)

# === Save updated metadata ===
all_metadata.to_csv(output_meta_path, index=False)
print(f"✅ Final metadata saved to {output_meta_path} ({len(all_metadata)} rows)")

# === Save updated images ===
with h5py.File(output_hdf5_path, "w") as hf:
    hf.create_dataset("images", data=all_images)

print(f"✅ Final HDF5 saved to {output_hdf5_path} ({len(all_images)} images)")
