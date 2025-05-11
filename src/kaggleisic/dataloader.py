import os

from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

BATCH_SIZE = 16  # batch size
NUM_SAMPLES = 500  # samples per epoch
NUM_WORKERS = 4  # number of CPU threads


def create_dataloaders(train_dataset, valid_dataset, test_dataset, sample_df):
    num_workers = min(NUM_WORKERS, os.cpu_count() or 1)
    print(f"Using {num_workers} CPU threads for data loading.")

    sampler = WeightedRandomSampler(
        weights=compute_sample_weights(sample_df),
        num_samples=NUM_SAMPLES,
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=num_workers,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
    )

    full_dataset = ConcatDataset([train_dataset, valid_dataset])
    full_loader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
    )

    print(
        f"Train loader: {len(train_loader)} batches (total = {NUM_SAMPLES} samples / {BATCH_SIZE} batches)"
    )
    print(f"Valid loader: {len(valid_loader)} batches")
    print(f"Test loader:  {len(test_loader)} batches")
    print(f"Full loader:  {len(full_loader)} batches")

    return train_loader, valid_loader, test_loader, full_loader


def compute_sample_weights(sample_df):
    # Compute sample weights based on the class distribution
    class_counts = sample_df["target"].value_counts().sort_index()
    class_weights = 1.0 / class_counts

    # Normalize weights to sum to 1
    class_weights = class_weights / class_weights.sum()

    sample_weights = sample_df["target"].map(class_weights).values

    return sample_weights
