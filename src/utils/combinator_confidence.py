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


def clinical_risk_tiering(paths, output_path="submission_clinical_tiers.csv"):
    if len(paths) < 2:
        raise ValueError("At least two file paths must be provided.")

    # Load and merge all dataframes on 'isic_id'
    dfs = []
    for i, path in enumerate(paths):
        df = pd.read_csv(path).copy()
        df.rename(columns={"target": f"target_{i+1}"}, inplace=True)
        dfs.append(df)

    # Merge on 'isic_id'
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="isic_id")

    # Columns with model predictions
    target_cols = [f"target_{i+1}" for i in range(len(paths))]

    # Define rule based on clinical tiering
    def clinical_rule(row):
        preds = row[target_cols]
        max_pred = preds.max()
        min_pred = preds.min()

        if max_pred >= 0.8:
            return max_pred  # High risk
        elif max_pred >= 0.5:
            return preds.mean()  # Medium risk
        else:
            return min_pred  # Low risk

    merged["target"] = merged.apply(clinical_rule, axis=1)

    # Save to CSV
    result = merged[["isic_id", "target"]]
    result.to_csv(output_path, index=False)
    print(f"Saved clinically tiered prediction to: {output_path}")


def consensus_escalation_heuristic(
    paths, output_path="submission_consensus_heuristic.csv"
):
    if len(paths) < 2:
        raise ValueError("At least two file paths must be provided.")

    # Load submissions
    dfs = []
    for i, path in enumerate(paths):
        df = pd.read_csv(path).copy()
        df.rename(columns={"target": f"target_{i+1}"}, inplace=True)
        dfs.append(df)

    # Merge on isic_id
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="isic_id")

    target_cols = [f"target_{i+1}" for i in range(len(paths))]

    def consensus_logic(row):
        preds = row[target_cols]
        num_high = (preds >= 0.7).sum()
        num_low = (preds <= 0.3).sum()
        std_dev = preds.std()
        min_pred, max_pred = preds.min(), preds.max()

        if num_high >= 3:
            return max_pred  # Strong consensus: high risk
        elif num_low >= 3:
            return min_pred  # Strong consensus: low risk
        elif std_dev >= 0.2:
            return preds.mean()  # Disagreement
        elif preds.between(0.3, 0.7).all():
            return preds.mean()  # All uncertain
        else:
            return preds.median()  # Fallback

    merged["target"] = merged.apply(consensus_logic, axis=1)

    # Output
    result = merged[["isic_id", "target"]]
    result.to_csv(output_path, index=False)
    print(f"Saved consensus heuristic prediction to: {output_path}")


# Aquí pones los paths de las submissions que quieras combinar
paths = [
    LGBM_CB_XGB_SUBMISSION_PATH,
    CONVNEXT_AUG_SUBMISSION_PATH,
    LGBM_SUBMISSION_PATH,
]

OUTPUT_SUBMISSION_PATH = urlparse(
    str(
        config.SUBMISSION_DATA_DIR
        / "ensemble/submission_mega_ensemble_hybrid_50_thresh.csv"
    )
).path

hybrid_worst_case_average(paths, output_path=OUTPUT_SUBMISSION_PATH)  # Aquí guardas


# ------- RESULTS
# hybrid_worst_case_average(paths, output_path=OUTPUT_SUBMISSION_PATH, threshold=0.6)
# Best one with: LGBM_CB_XGB_SUBMISSION_PATH, CONVNEXT_AUG_SUBMISSION_PATH, LGBM_SUBMISSION_PATH and threshold 0.6
#################
#
