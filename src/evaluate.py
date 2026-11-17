import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from typing import Tuple, List


from dataset import SurfaceDefectDataset, get_transforms
from model import get_model

def get_test_paths_and_labels(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Scans the test directory to extract image paths and labels.
    """
    base_path = Path(data_dir)
    image_paths = []
    labels = []
    
    ok_dir = base_path / "ok_front"
    if ok_dir.exists():
        for img_path in ok_dir.glob("*.jpeg"):
            image_paths.append(str(img_path))
            labels.append(0)
            
    def_dir = base_path / "def_front"
    if def_dir.exists():
        for img_path in def_dir.glob("*.jpeg"):
            image_paths.append(str(img_path))
            labels.append(1)
            
    return image_paths, labels

def evaluate_model():
    """
    Evaluates the trained model on the test dataset.
    Calculates Accuracy, Precision, Recall, and the Confusion Matrix.
    """
    # Configuration & Setup
    test_dir = "data/raw/casting_data/test"
    model_weights_path = "models/best_resnet18.pth"
    batch_size = 32
    num_classes = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluation Device: {device}")


    # Load Test Data
    test_paths, test_labels = get_test_paths_and_labels(test_dir)
    if not test_paths:
        raise ValueError(f"No test images found in {test_dir}.")

    test_dataset = SurfaceDefectDataset(
        image_paths=test_paths,
        labels=test_labels,
        transform=get_transforms(is_train=False)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the Trained Model
    model = get_model(num_classes=num_classes, fine_tune=False)
    
    try:
        model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
        print(f"[*] Weights loaded from: {model_weights_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Weights file not found at {model_weights_path}.")

    model = model.to(device)
    model.eval()

    # Inference (Make Predictions)
    all_true_labels = []
    all_predictions = []

    print("[*] Running inference...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_true_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Quality Analytics (Metrics & Confusion Matrix)
    print("\n--- EVALUATION RESULTS ---")
    
    target_names = ['OK (Class 0)', 'NG (Class 1)']
    print(classification_report(all_true_labels, all_predictions, target_names=target_names))

    cm = confusion_matrix(all_true_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()

    print("\n--- CONFUSION MATRIX ---")
    print(f"Total Images: {len(all_true_labels)}")
    print(f"TN (True Negative)  : {tn}")
    print(f"TP (True Positive)  : {tp}")
    print(f"FP (False Positive) : {fp}")
    print(f"FN (False Negative) : {fn}")
    print("-" * 25)

if __name__ == "__main__":
    evaluate_model()