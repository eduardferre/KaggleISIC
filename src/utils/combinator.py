from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv

from kaggleisic import config

load_dotenv()

IMG_SUBMISSION_PATH = config.SUBMISSION_DATA_DIR / "convnext/convnext_aug_hdf5.csv"
LGBM_SUBMISSION_PATH = config.SUBMISSION_DATA_DIR / "tabular/lgbm_submission.csv"
OUTPUT_SUBMISSION_PATH = urlparse(
    str(config.SUBMISSION_DATA_DIR / "ensemble/submission_ensemble_avg.csv")
).path

# Load the submissions
img_df = pd.read_csv(IMG_SUBMISSION_PATH)
meta_lgbm_df = pd.read_csv(LGBM_SUBMISSION_PATH)

# Merge on isic_id
merged_lgbm = pd.merge(img_df, meta_lgbm_df, on="isic_id", suffixes=("_img", "_meta"))

# Average the MGMT_values
merged_lgbm["target"] = (merged_lgbm["target_img"] + merged_lgbm["target_meta"]) / 2

# Keep only required columns
final_submission_lgbm = merged_lgbm[["isic_id", "target"]]

# Save to new CSV
final_submission_lgbm.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
