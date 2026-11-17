
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from pathlib import Path
from typing import Tuple, List

from dataset import SurfaceDefectDataset, get_transforms
from model import get_model


def get_image_paths_and_labels(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Scans the directory structure to extract image paths and their corresponding labels.
    """
    base_path = Path(data_dir)
    image_paths = []
    labels = []

    # Label 0: OK products
    ok_dir = base_path / "ok_front"
    if ok_dir.exists():
        for img_path in ok_dir.glob("*.jpeg"):
            image_paths.append(str(img_path))
            labels.append(0)

    # Label 1: Defective products (NG)
    def_dir = base_path / "def_front"
    if def_dir.exists():
        for img_path in def_dir.glob("*.jpeg"):
            image_paths.append(str(img_path))
            labels.append(1)

    return image_paths, labels


def train_model():
    """
    Main function to train the Surface Defect Detection model.
    """

    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    num_classes = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Data Preparation
    train_dir = "data/raw/casting_data/train"
    test_dir = "data/raw/casting_data/test"

    train_paths, train_labels = get_image_paths_and_labels(train_dir)
    val_paths, val_labels = get_image_paths_and_labels(test_dir)

    print(f"    -> Found {len(train_paths)} training images.")
    print(f"    -> Found {len(val_paths)} validation images.")

    if len(train_paths) == 0 or len(val_paths) == 0:
        raise ValueError(
            "Dataset not found! Please check your folder structure and .jpeg extension."
        )

    train_dataset = SurfaceDefectDataset(
        image_paths=train_paths,
        labels=train_labels,
        transform=get_transforms(is_train=True)
    )

    val_dataset = SurfaceDefectDataset(
        image_paths=val_paths,
        labels=val_labels,
        transform=get_transforms(is_train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Model, Loss, and Optimizer Setup
    model = get_model(num_classes=num_classes, fine_tune=False)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0

    # Training Loop

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        # TRAINING PHASE
        model.train()

        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_epoch_loss = running_loss / total_train
        train_epoch_acc = (correct_train / total_train) * 100

        # VALIDATION PHASE
        model.eval()

        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)

                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / total_val
        val_epoch_acc = (correct_val / total_val) * 100

        print(f"Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc:.2f}%")
        print(f"Val Loss:   {val_epoch_loss:.4f} | Val Acc:   {val_epoch_acc:.2f}%")

        # 5. Save the Best Model
        if val_epoch_acc > best_val_accuracy:
            best_val_accuracy = val_epoch_acc

            os.makedirs("models", exist_ok=True)

            save_path = "models/best_resnet18.pth"
            torch.save(model.state_dict(), save_path)

            print(
                f"[*] New best model saved to {save_path} "
                f"with accuracy: {best_val_accuracy:.2f}%"
            )


if __name__ == "__main__":
    train_model()
