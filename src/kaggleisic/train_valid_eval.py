from urllib.parse import urlparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm

from kaggleisic import config

EPOCHS = 1
LEARNING_RATE = 1e-4
SCHEDULER_STEP_SIZE = 2
SCHEDULER_GAMMA = 0.1

load_dotenv()


def train_valid(model, train_loader, valid_loader, is_multimodal=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    # Tracking lists
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(1, EPOCHS + 1):
        if is_multimodal:
            train_acc = train_multimodal(
                model, device, train_loader, optimizer, criterion, epoch
            )
            valid_acc = validate_multimodal(
                model, device, valid_loader, criterion, epoch
            )
        else:
            train_acc = train_singles(
                model, device, train_loader, optimizer, criterion, epoch
            )
            valid_acc = validate_singles(model, device, valid_loader, criterion, epoch)

        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate: {current_lr}")

    # Plot training and validation accuracies
    plot_train_valid_curves(train_accuracies, valid_accuracies)

    print("Training complete ✅")


def train_eval(
    model,
    full_loader,
    test_loader,
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
    train_accuracies = []

    for epoch in range(1, EPOCHS + 1):
        if is_multimodal:
            train_acc = train_multimodal(
                model, device, full_loader, optimizer, criterion, epoch
            )
        else:
            train_acc = train_singles(
                model, device, full_loader, optimizer, criterion, epoch
            )
        train_accuracies.append(train_acc)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate: {current_lr}")

    # Save final model
    output_model_path = urlparse(str(config.MODELS_DATA_DIR / output_model_file)).path
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")

    # Plot training and validation accuracies
    plot_train_curves(train_accuracies)

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


def train_multimodal(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

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
        predicted = (logits >= 0.5).float()
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = correct_preds / total_preds

    print(
        f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}"
    )
    return train_accuracy


def validate_multimodal(model, device, valid_loader, criterion, epoch):
    model.eval()
    val_loss = 0.0
    val_correct_preds = 0
    val_total_preds = 0

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
            predicted = (logits >= 0.5).float()
            val_correct_preds += (predicted == labels).sum().item()
            val_total_preds += labels.size(0)

    avg_val_loss = val_loss / len(valid_loader)
    val_accuracy = val_correct_preds / val_total_preds
    print(
        f"Epoch {epoch}/{EPOCHS} | Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}"
    )
    return val_accuracy


def evaluate_multimodal(model, device, test_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for metadatas, images, isic_ids in tqdm(test_loader, desc="Inference on Test"):
            metadatas, images = metadatas.to(device).float(), images.to(device)

            logits = model(images, metadatas).view(-1)  # shape [batch_size]
            probs = torch.sigmoid(logits)  # shape [batch_size], in [0,1]

            probs = probs.cpu().numpy()

            for isic_id, p in zip(isic_ids, probs):
                predictions.append({"isic_id": isic_id, "target": float(p)})

    submission_df = pd.DataFrame(predictions)
    submission_df = submission_df.sort_values(by="isic_id").reset_index(drop=True)

    return submission_df


def train_singles(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for singles, labels, _ in tqdm(
        train_loader, desc=f"Train Epoch {epoch}", leave=False
    ):
        singles, labels = singles.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(singles).view(-1)  # [batch_size]

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (logits >= 0.5).float()
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = correct_preds / total_preds

    print(
        f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}"
    )
    return train_accuracy


def validate_singles(model, device, valid_loader, criterion, epoch):
    model.eval()
    val_loss = 0.0
    val_correct_preds = 0
    val_total_preds = 0

    with torch.no_grad():
        for singles, labels, _ in tqdm(
            valid_loader, desc=f"Validation Epoch {epoch}", leave=False
        ):
            singles, labels = singles.to(device), labels.to(device)

            logits = model(singles).view(-1)
            loss = criterion(logits, labels)

            val_loss += loss.item()
            predicted = (logits >= 0.5).float()
            val_correct_preds += (predicted == labels).sum().item()
            val_total_preds += labels.size(0)

    avg_val_loss = val_loss / len(valid_loader)
    val_accuracy = val_correct_preds / val_total_preds
    print(
        f"Epoch {epoch}/{EPOCHS} | Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}"
    )
    return val_accuracy


def evaluate_singles(model, device, test_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for singles, isic_ids in tqdm(test_loader, desc="Inference on Test"):
            singles = singles.to(device)

            logits = model(singles).view(-1)  # shape [batch_size]
            probs = torch.sigmoid(logits)  # shape [batch_size], in [0,1]

            probs = probs.cpu().numpy()

            for isic_id, p in zip(isic_ids, probs):
                predictions.append({"isic_id": isic_id, "target": float(p)})

    submission_df = pd.DataFrame(predictions)
    submission_df = submission_df.sort_values(by="isic_id").reset_index(drop=True)

    return submission_df


def plot_train_valid_curves(train_accs, val_accs):
    # Prepare DataFrame for seaborn
    epochs = list(range(1, len(train_accs) + 1))
    data = pd.DataFrame(
        {
            "Epoch": epochs * 2,
            "Accuracy": train_accs + val_accs,
            "Phase": ["Train"] * len(train_accs) + ["Validation"] * len(val_accs),
        }
    )

    # Plot with seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="Epoch", y="Accuracy", hue="Phase", marker="o")
    plt.title("Training vs Validation Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def plot_train_curves(train_accs):
    # Prepare DataFrame for seaborn
    epochs = list(range(1, len(train_accs) + 1))
    data = pd.DataFrame(
        {
            "Epoch": epochs,
            "Accuracy": train_accs,
        }
    )

    # Plot with seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="Epoch", y="Accuracy", marker="o")
    plt.title("Training Accuracy (Full Dataset)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
