from urllib.parse import urlparse

import pandas as pd
import torch
import torchvision.transforms as T
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from kaggleisic import config
from kaggleisic.isic_class import ISIC_HDF5_Dataset, ISIC_Multimodal_Dataset

TRAIN_METADATA_CSV = "new-train-metadata.csv"
TEST_METADATA_CSV = "students-test-metadata.csv"
TRAIN_METADATA_PROCESSED_CSV = "train-metadata-processed.csv"
TEST_METADATA_PROCESSED_CSV = "test-metadata-processed.csv"
TRAIN_HDF5 = urlparse(str(config.RAW_DATA_DIR / "train-image.hdf5")).path
TEST_HDF5 = urlparse(str(config.RAW_DATA_DIR / "test-image.hdf5")).path
TRAIN_METADATA_AUGMENTED_CSV = "train-metadata-augmented.csv"
TRAIN_HDF5_AUGMENTED = urlparse(
    str(config.PROCESSED_DATA_DIR / "train-image-augmented.hdf5")
).path

DROP_COLUMNS = [
    "image_type",
    "patient_id",
    "copyright_license",
    "attribution",
    "anatom_site_general",
    "tbp_lv_location_simple",
]

load_dotenv()


def process_metadata() -> tuple:
    """
    Process the metadata CSV files by encoding categorical features and imputing missing values.
    Returns:
        tuple: A tuple containing the train and test datasets.
    """
    # Load the metadata CSV files
    train_df = pd.read_csv(config.RAW_DATA_DIR / TRAIN_METADATA_CSV)
    test_df = pd.read_csv(config.RAW_DATA_DIR / TEST_METADATA_CSV)

    print(f"train_df shape: {train_df.shape}")
    print(f"test_df shape:  {test_df.shape}")

    # Check matching columns and drop non-matching columns
    shared_columns = set(train_df.columns).intersection(set(test_df.columns))
    shared_columns.add("target")  # Ensure 'target' is included
    train_df = train_df[list(shared_columns)]

    train_df = train_df.drop(columns=DROP_COLUMNS)
    test_df = test_df.drop(columns=DROP_COLUMNS)

    print(f"train_df shape after dropping columns: {train_df.shape}")
    print(f"test_df shape after dropping columns:  {test_df.shape}")

    # Encode and impute features
    train_df = encode_impute_data(train_df)
    test_df = encode_impute_data(test_df)

    print(f"train_df shape after encoding: {train_df.shape}")
    print(f"test_df shape after encoding:  {test_df.shape}")

    train_dataset = train_df.reset_index(drop=True)
    test_dataset = test_df.reset_index(drop=True)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    return train_dataset, test_dataset


def encode_impute_data(df_metadata, exclude_cols=["target", "isic_id"]) -> pd.DataFrame:
    """
    Encode and impute missing values in the metadata DataFrame.
    Args:
        df_metadata (pd.DataFrame): The metadata DataFrame.
        exclude_cols (list): List of columns to exclude from imputation.
    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """
    from concurrent.futures import ThreadPoolExecutor

    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import LabelEncoder

    df = df_metadata.copy()

    features = [col for col in df.columns if col not in exclude_cols]
    cat_cols = df[features].select_dtypes(include=["object"]).columns.tolist()

    # Store valid label values per categorical column
    valid_label_values = {}

    def encode_column(col):
        le = LabelEncoder()
        non_null_mask = df[col].notnull()
        df.loc[non_null_mask, col] = le.fit_transform(
            df.loc[non_null_mask, col].astype(str)
        )
        valid_label_values[col] = list(le.transform(le.classes_))

    # Parallel label encoding
    with ThreadPoolExecutor() as executor:
        executor.map(encode_column, cat_cols)

    # Define features with missing values
    features_with_missing = df[features].columns[df[features].isnull().any()].tolist()

    # Impute missing values using KNN only for missing values rows
    imputer = KNNImputer(n_neighbors=5)
    df[features_with_missing] = imputer.fit_transform(df[features_with_missing])

    # Post-process imputed categorical columns
    for col in cat_cols:
        if col in features_with_missing:
            # Map float to nearest valid class
            df[col] = df[col].apply(
                lambda x: min(valid_label_values[col], key=lambda v: abs(v - x))
            )

    return df


def load_metadata_dataset(train_frac=0.8, seed=42, is_augmented=False) -> tuple:
    if is_augmented:
        train_file = TRAIN_METADATA_AUGMENTED_CSV
    else:
        train_file = TRAIN_METADATA_PROCESSED_CSV
    # Load the metadata CSV files
    train_df = pd.read_csv(config.PROCESSED_DATA_DIR / train_file)
    test_df = pd.read_csv(config.PROCESSED_DATA_DIR / TEST_METADATA_PROCESSED_CSV)

    # Perform stratified train/validation split to maintain class distribution
    train_dataset, valid_dataset = train_test_split(
        train_df, train_size=train_frac, stratify=train_df["target"], random_state=seed
    )

    # Reset index for train and validation datasets
    train_dataset = train_dataset.reset_index(drop=True)
    valid_dataset = valid_dataset.reset_index(drop=True)
    test_dataset = test_df.reset_index(drop=True)

    print(f"train_dataset shape: {train_dataset.shape}")
    print(f"valid_dataset shape: {valid_dataset.shape}")
    print(f"test_dataset shape:  {test_dataset.shape}")

    return train_dataset, valid_dataset, test_dataset


def load_hdf5_dataset(
    transform: T.Compose, train_frac=0.8, seed=42, is_augmented=False
) -> tuple[ISIC_HDF5_Dataset]:
    """
    Load the ISIC dataset from HDF5 files and split it into train, validation, and test sets.
    Args:
        transform (T.Compose): Transformations to apply to the images.
        train_frac (float): Fraction of the dataset to use for training.
        seed (int): Random seed for reproducibility.
    Returns:
        tuple: A tuple containing the train, validation, and test datasets.
    """
    # Load the metadata CSV files
    train_df_sub, valid_df_sub, test_df = load_metadata_dataset(
        train_frac=train_frac, seed=seed, is_augmented=is_augmented
    )

    if is_augmented:
        train_file = TRAIN_HDF5_AUGMENTED
    else:
        train_file = TRAIN_HDF5

    # Create Datasets
    train_dataset = ISIC_HDF5_Dataset(
        df=train_df_sub, hdf5_path=train_file, transform=transform, is_labelled=True
    )

    valid_dataset = ISIC_HDF5_Dataset(
        df=valid_df_sub, hdf5_path=train_file, transform=transform, is_labelled=True
    )

    test_dataset = ISIC_HDF5_Dataset(
        df=test_df, hdf5_path=TEST_HDF5, transform=transform, is_labelled=False
    )

    return train_dataset, valid_dataset, test_dataset


def load_multimodal_dataset(
    transform: T.Compose, train_frac=0.8, seed=42, is_augmented=False
) -> tuple[ISIC_Multimodal_Dataset]:
    """
    Load the ISIC dataset from HDF5 files and split it into train, validation, and test sets.
    Args:
        transform (T.Compose): Transformations to apply to the images.
        train_frac (float): Fraction of the dataset to use for training.
        seed (int): Random seed for reproducibility.
    Returns:
        tuple: A tuple containing the train, validation, and test datasets.
    """
    # Load the metadata CSV files
    train_df_sub, valid_df_sub, test_df = load_metadata_dataset(
        train_frac=train_frac, seed=seed, is_augmented=is_augmented
    )

    if is_augmented:
        train_file = TRAIN_HDF5_AUGMENTED
    else:
        train_file = TRAIN_HDF5

    # Create Datasets
    train_dataset = ISIC_Multimodal_Dataset(
        df=train_df_sub,
        hdf5_path=train_file,
        transform=transform,
        is_labelled=True,
    )

    valid_dataset = ISIC_Multimodal_Dataset(
        df=valid_df_sub,
        hdf5_path=train_file,
        transform=transform,
        is_labelled=True,
    )

    test_dataset = ISIC_Multimodal_Dataset(
        df=test_df,
        hdf5_path=TEST_HDF5,
        transform=transform,
        is_labelled=False,
    )

    return train_dataset, valid_dataset, test_dataset


def create_balanced_subset(
    df: pd.DataFrame, target_col="target", seed=42
) -> pd.DataFrame:
    # Just keep all the cancer cases and subsample the healthy cases (2:1 ratio)
    positives = df[df[target_col] == 1]

    n_negatives = len(positives) * 2  # 2:1 ratio
    negatives = df[df[target_col] == 0].sample(
        n=min(n_negatives, len(df[df[target_col] == 0])), random_state=seed
    )
    balanced_df = (
        pd.concat([positives, negatives])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    return balanced_df
