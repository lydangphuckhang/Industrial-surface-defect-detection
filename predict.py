import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "src"))

from src.preprocessing import load_image, preprocess_image
from src.dataset import get_transforms
from src.model import get_model

def predict_single_image(image_path: str, model_weights: str = "models/best_resnet18.pth"):
    """
    Runs end-to-end inference on a single industrial image.
    Outputs the prediction (OK/NG) and the confidence score.
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=2, fine_tune=False)
    
    try:
        model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=True))
    except FileNotFoundError:
        raise FileNotFoundError(f"Model weights missing at {model_weights}. Train the model first.")
        
    model = model.to(device)
    model.eval() # Disable dropout/batchnorm for inference

    # Pipeline: Image Processing (OpenCV)
    image_np = load_image(image_path, as_gray=False)
    image_preprocessed = preprocess_image(image_np, size=224)

    # PyTorch Transformations (Normalization)
    # Use is_train=False to avoid data augmentation like random flips
    transform_pipeline = get_transforms(is_train=False)
    image_tensor = transform_pipeline(image_preprocessed)
    
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Inference and Confidence Calculation
    with torch.no_grad():
        outputs = model(image_tensor)
        
        # Apply Softmax to convert logits to probabilities [0.0, 1.0]
        probabilities = F.softmax(outputs, dim=1)
        
        # Get the predicted class index (0 for OK, 1 for NG)
        confidence_score, predicted_class = torch.max(probabilities, 1)

    # Format Output
    class_names = {0: "OK (Passed)", 1: "NG (Defective)"}
    result = class_names[predicted_class.item()]
    confidence = confidence_score.item() * 100

    print("-" * 40)
    print(" INFERENCE RESULT")
    print("-" * 40)
    print(f"Image      : {image_path}")
    print(f"Prediction : {result}")
    print(f"Confidence : {confidence:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Surface Defect (OK/NG)")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--weights", type=str, default="models/best_resnet18.pth", help="Path to model weights")
    
    args = parser.parse_args()
    predict_single_image(args.image, args.weights)