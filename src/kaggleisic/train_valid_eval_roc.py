from urllib.parse import urlparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm

from kaggleisic import config

EPOCHS = 1
LEARNING_RATE = 1e-4
SCHEDULER_STEP_SIZE = 2
SCHEDULER_GAMMA = 0.1
MIN_DELTA = 0.001

load_dotenv()


def train_valid(
    model, train_loader, valid_loader, patience=5, is_multimodal=False
) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    # Tracking lists
    train_aucs = []
    valid_aucs = []

    best_valid_auc = 0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        if is_multimodal:
            train_auc = train_multimodal(
                model, device, train_loader, optimizer, criterion, epoch
            )
            valid_auc = validate_multimodal(
                model, device, valid_loader, criterion, epoch
            )
        else:
            train_auc = train_singles(
                model, device, train_loader, optimizer, criterion, epoch
            )
            valid_auc = validate_singles(model, device, valid_loader, criterion, epoch)

        train_aucs.append(train_auc)
        valid_aucs.append(valid_auc)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate: {current_lr}")

        # Early Stopping check with threshold
        if valid_auc > best_valid_auc + MIN_DELTA:
            best_valid_auc = valid_auc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(
                f"No improvement in {epochs_no_improve} epochs (threshold of {MIN_DELTA})."
            )

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs!")
            break

    # Plot training and validation ROC AUC scores
    plot_train_valid_curves(train_aucs, valid_aucs)

    print("Training complete ✅")

    return epoch


def train_eval(
    model,
    full_loader,
    test_loader,
    early_stopping_epochs=EPOCHS,
    is_multimodal=False,
    output_model_file="model_final.pth",
    output_submission_file="submission.csv",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    # Tracking lists
    train_aucs = []

    for epoch in range(1, early_stopping_epochs + 1):
        if is_multimodal:
            train_auc = train_multimodal(
                model, device, full_loader, optimizer, criterion, epoch
            )
        else:
            train_auc = train_singles(
                model, device, full_loader, optimizer, criterion, epoch
            )
        train_aucs.append(train_auc)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate: {current_lr}")

    # Save final model
    output_model_path = urlparse(str(config.MODELS_DATA_DIR / output_model_file)).path
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")

    # Plot training ROC AUC scores
    plot_train_curves(train_aucs)

    print("Training complete ✅")

    # Evaluate on test set
    if is_multimodal:
        submission_df = evaluate_multimodal(model, device, test_loader)
    else:
        submission_df = evaluate_singles(model, device, test_loader)

    # Save submission file
    submission_file_path = urlparse(
        str(config.SUBMISSION_DATA_DIR / output_submission_file)
    ).path
    submission_df.to_csv(submission_file_path, index=False)

    print(
        f"Saved submission with {len(submission_df)} rows to {submission_file_path} ✅"
    )


def train_singles(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    all_logits = []
    all_labels = []

    for singles, labels, _ in tqdm(
        train_loader, desc=f"Train Epoch {epoch}", leave=False
    ):
        singles, labels = singles.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(singles).view(-1)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_logits.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_train_loss = running_loss / len(train_loader)
    try:
        train_auc = roc_auc_score(all_labels, all_logits)
    except ValueError:
        train_auc = 0.0

    print(
        f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train ROC AUC: {train_auc:.4f}"
    )
    return train_auc


def validate_singles(model, device, valid_loader, criterion, epoch):
    model.eval()
    val_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for singles, labels, _ in tqdm(
            valid_loader, desc=f"Validation Epoch {epoch}", leave=False
        ):
            singles, labels = singles.to(device), labels.to(device)

            logits = model(singles).view(-1)
            loss = criterion(logits, labels)

            val_loss += loss.item()
            all_logits.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(valid_loader)
    try:
        val_auc = roc_auc_score(all_labels, all_logits)
    except ValueError:
        val_auc = 0.0

    print(
        f"Epoch {epoch}/{EPOCHS} | Validation Loss: {avg_val_loss:.4f} | Validation ROC AUC: {val_auc:.4f}"
    )
    return val_auc


def evaluate_singles(model, device, test_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for singles, isic_ids in tqdm(test_loader, desc="Inference on Test"):
            singles = singles.to(device)

            logits = model(singles).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

            for isic_id, p in zip(isic_ids, probs):
                predictions.append({"isic_id": isic_id, "target": float(p)})

    submission_df = pd.DataFrame(predictions)
    submission_df = submission_df.sort_values(by="isic_id").reset_index(drop=True)

    return submission_df


def train_multimodal(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    all_logits = []
    all_labels = []

    for metadatas, images, labels in tqdm(
        train_loader, desc=f"Train Epoch {epoch}", leave=False
    ):
        metadatas, images, labels = (
            metadatas.to(device).float(),
            images.to(device),
            labels.to(device),
        )

        optimizer.zero_grad()
        logits = model(images, metadatas).view(-1)  # [batch_size]

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_logits.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_train_loss = running_loss / len(train_loader)
    try:
        train_auc = roc_auc_score(all_labels, all_logits)
    except ValueError:
        train_auc = 0.0

    print(
        f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train ROC AUC: {train_auc:.4f}"
    )
    return train_auc


def validate_multimodal(model, device, valid_loader, criterion, epoch):
    model.eval()
    val_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for metadatas, images, labels in tqdm(
            valid_loader, desc=f"Validation Epoch {epoch}", leave=False
        ):
            metadatas, images, labels = (
                metadatas.to(device).float(),
                images.to(device),
                labels.to(device),
            )

            logits = model(images, metadatas).view(-1)
            loss = criterion(logits, labels)

            val_loss += loss.item()
            all_logits.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(valid_loader)
    try:
        val_auc = roc_auc_score(all_labels, all_logits)
    except ValueError:
        val_auc = 0.0

    print(
        f"Epoch {epoch}/{EPOCHS} | Validation Loss: {avg_val_loss:.4f} | Validation ROC AUC: {val_auc:.4f}"
    )
    return val_auc


def evaluate_multimodal(model, device, test_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for metadatas, images, isic_ids in tqdm(test_loader, desc="Inference on Test"):
            metadatas, images = metadatas.to(device).float(), images.to(device)

            logits = model(images, metadatas).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

            for isic_id, p in zip(isic_ids, probs):
                predictions.append({"isic_id": isic_id, "target": float(p)})

    submission_df = pd.DataFrame(predictions)
    submission_df = submission_df.sort_values(by="isic_id").reset_index(drop=True)

    return submission_df


def plot_train_valid_curves(train_aucs, val_aucs):
    # Prepare DataFrame for seaborn
    epochs = list(range(1, len(train_aucs) + 1))
    data = pd.DataFrame(
        {
            "Epoch": epochs * 2,
            "ROC AUC": train_aucs + val_aucs,
            "Phase": ["Train"] * len(train_aucs) + ["Validation"] * len(val_aucs),
        }
    )

    # Plot with seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="Epoch", y="ROC AUC", hue="Phase", marker="o")
    plt.title("Training vs Validation ROC AUC")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def plot_train_curves(train_aucs):
    # Prepare DataFrame for seaborn
    epochs = list(range(1, len(train_aucs) + 1))
    data = pd.DataFrame(
        {
            "Epoch": epochs,
            "ROC AUC": train_aucs,
        }
    )

    # Plot with seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="Epoch", y="ROC AUC", marker="o")
    plt.title("Training ROC AUC (Full Dataset)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
