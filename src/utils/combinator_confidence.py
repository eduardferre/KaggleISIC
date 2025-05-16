from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv

from kaggleisic import config

load_dotenv()

LGBM_SUBMISSION_PATH = config.SUBMISSION_DATA_DIR / "tabular/lgbm_submission.csv"
LGBM_CB_XGB_SUBMISSION_PATH = config.SUBMISSION_DATA_DIR / "tabular/lgbm_cb_xgb.csv"
CONVNEXT_AUG_SUBMISSION_PATH = (
    config.SUBMISSION_DATA_DIR / "convnext/convnext_aug_hdf5.csv"
)
CONVNEXT_MORE_IMAGES_SUBMISSION_PATH = (
    config.SUBMISSION_DATA_DIR / "convnext/convnext_more_images.csv"
)
RESNET_SIMP_AUG_SUBMISSION_PATH = (
    config.SUBMISSION_DATA_DIR / "resnet/resnet_simp_aug_hdf5.csv"
)


def confidence_voting(paths, output_path="submission_confidence_voting.csv"):

    if len(paths) < 2:
        raise ValueError("At least two file paths must be provided.")

    # Load and merge all dataframes on 'isic_id'
    dfs = []
    for i, path in enumerate(paths):
        df = pd.read_csv(path).copy()
        df.rename(columns={"target": f"target_{i+1}"}, inplace=True)
        dfs.append(df)

    # Merge all on 'isic_id'
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="isic_id")

    # Select the most confident prediction per row
    target_cols = [f"target_{i+1}" for i in range(len(paths))]
    merged["target"] = merged[target_cols].apply(
        lambda row: row.loc[row.sub(0.5).abs().idxmax()], axis=1
    )

    # Output result
    result = merged[["isic_id", "target"]]
    result.to_csv(output_path, index=False)
    print(f"Saved most confident prediction to: {output_path}")


def worst_case_voting(paths, output_path="submission_worst_case_voting.csv"):
    if len(paths) < 2:
        raise ValueError("At least two file paths must be provided.")

    # Load and merge all dataframes on 'isic_id'
    dfs = []
    for i, path in enumerate(paths):
        df = pd.read_csv(path).copy()
        df.rename(columns={"target": f"target_{i+1}"}, inplace=True)
        dfs.append(df)

    # Merge all on 'isic_id'
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="isic_id")

    # Select the worst-case prediction per row (maximum probability)
    target_cols = [f"target_{i+1}" for i in range(len(paths))]
    merged["target"] = merged[target_cols].max(axis=1)

    # Output result
    result = merged[["isic_id", "target"]]
    result.to_csv(output_path, index=False)
    print(f"Saved worst-case (maximum) prediction to: {output_path}")


def hybrid_worst_case_average(
    paths, output_path="submission_hybrid_voting.csv", threshold=0.6
):
    if len(paths) < 2:
        raise ValueError("At least two file paths must be provided.")

    # Load and merge all dataframes on 'isic_id'
    dfs = []
    for i, path in enumerate(paths):
        df = pd.read_csv(path).copy()
        df.rename(columns={"target": f"target_{i+1}"}, inplace=True)
        dfs.append(df)

    # Merge all on 'isic_id'
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="isic_id")

    # Columns with predictions
    target_cols = [f"target_{i+1}" for i in range(len(paths))]

    # Apply hybrid rule
    def hybrid_rule(row):
        preds = row[target_cols]
        if preds.max() >= threshold:
            return preds.max()  # Worst-case: maximum if any prediction is confident
        else:
            return preds.mean()  # Otherwise, average

    merged["target"] = merged.apply(hybrid_rule, axis=1)

    # Save result
    result = merged[["isic_id", "target"]]
    result.to_csv(output_path, index=False)
    print(f"Saved hybrid prediction to: {output_path}")


# Aquí pones los paths de las submissions que quieras combinar
paths = [
    LGBM_CB_XGB_SUBMISSION_PATH,
    CONVNEXT_AUG_SUBMISSION_PATH,
    LGBM_SUBMISSION_PATH,
]

OUTPUT_SUBMISSION_PATH = urlparse(
    str(
        config.SUBMISSION_DATA_DIR
        / "ensemble/submission_mega_ensemble_hybrid_approach_v3.csv"
    )
).path

hybrid_worst_case_average(paths, output_path=OUTPUT_SUBMISSION_PATH)  # Aquí guardas
