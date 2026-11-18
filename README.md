# Industrial Surface Defect Detection

An end-to-end prototype for automated visual inspection in industrial manufacturing using Computer Vision and Deep Learning.

## Motivation

In industrial manufacturing, surface quality inspection plays a critical role in ensuring product quality and reliability. Traditional visual inspection, which relies on human operators, can be time-consuming and subject to human error, particularly due to fatigue and inconsistent judgment.

This project explores the application of Computer Vision and Deep Learning to automate the detection and classification of surface defects in industrial products.

Using a Convolutional Neural Network (CNN) with Transfer Learning, the objective is to investigate how AI-based image classification can support industrial quality control processes by improving inspection consistency and helping reduce the risk of defective products reaching customers.

## How It Works

The project follows a modular pipeline combining classical Computer Vision techniques with Deep Learning:

1. **Image Input:** Takes surface images of manufactured casting products as input.
2. **Image Preprocessing (OpenCV):** Processes and prepares images before they enter the training and inference pipeline (`src/preprocessing.py`).
3. **Data Pipeline (PyTorch):** Implements a custom Dataset to load images, convert them into tensors, apply ImageNet normalization, and perform data augmentation (random horizontal and vertical flips) during training (`src/dataset.py`).
4. **Deep Learning Model:** Uses a pre-trained **ResNet18** architecture through Transfer Learning. The backbone is frozen (Feature Extraction), while a custom classification head is trained to distinguish between two classes: **OK (Non-defective)** and **NG (Defective)** (`src/model.py`).
5. **Model Evaluation:** Evaluates performance on the test set using a Confusion Matrix and classification metrics, with particular attention to False Positives and False Negatives from a Quality Engineering perspective (`src/evaluate.py`).

## Repository Structure

```text
industrial-surface-defect/
├── data/
│   └── raw/
│       └── casting_data/       # Dataset (not tracked by Git)
│
├── models/                     # Saved model weights (not tracked by Git)
│
├── notebooks/
│   └── image_preprocessing.ipynb
│
├── src/
│   ├── preprocessing.py        # OpenCV image preprocessing
│   ├── dataset.py              # PyTorch Dataset and transformations
│   ├── model.py                # ResNet18 model architecture
│   ├── train.py                # Training and validation pipeline
│   └── evaluate.py             # Model evaluation and quality metrics
│
├── predict.py                  # Single-image inference script
├── requirements.txt            # Python dependencies
└── README.md
```

## Installation and Usage

### 1. Environment Setup

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/industrial-surface-defect.git

cd industrial-surface-defect

# Create a virtual environment
python -m venv venv

# Activate the environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Main dependencies:**
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

### 2. Dataset Preparation

This project uses the **Casting Product Image Data for Quality Inspection** dataset.

1. Download the dataset from Kaggle.
2. Extract the dataset into `data/raw/casting_data/`.

The expected directory structure is:

```text
data/raw/casting_data/
├── train/
│   ├── ok_front/
│   └── def_front/
│
└── test/
    ├── ok_front/
    └── def_front/
```

### 3. Model Training

Train the ResNet18 model:

```bash
python src/train.py
```

The training script performs model training and validation, then saves the best-performing model weights to:

```text
models/best_resnet18.pth
```

### 4. Model Evaluation

Evaluate the trained model on the test dataset:

```bash
python src/evaluate.py
```

The evaluation script generates a Confusion Matrix and reports classification metrics to assess model performance.

### 5. Single Image Prediction

Run inference on a new image:

```bash
python predict.py --image "data/raw/casting_data/test/def_front/cast_def_0_110.jpeg"
```

The script returns the predicted class and its associated confidence score.

## Results and Quality Analysis

The model was evaluated on a test set of **715 images**, achieving approximately **99.02% accuracy**.

### Confusion Matrix Summary

| Metric | Count |
|---|---:|
| True Negatives (TN) | 259 |
| True Positives (TP) | 449 |
| False Positives (FP) | 3 |
| False Negatives (FN) | 4 |
| **Total** | **715** |

*Note: NG (Defective) is considered the positive class.*

### Quality Engineering Perspective

In industrial quality control, classification errors do not have the same business impact.

**1. False Positives (FP): 3**

Non-defective products (OK) incorrectly classified as defective (NG).

- **Business impact:** Unnecessary rejection of acceptable products, potentially increasing scrap costs and requiring additional manual inspection or rework.
- **Potential consequence:** Reduced production efficiency.

**2. False Negatives (FN): 4**

Defective products (NG) incorrectly classified as acceptable (OK).

- **Business impact:** Defective products may pass inspection and reach customers.
- **Potential consequence:** Increased customer complaints, warranty costs, and quality risks.

### Further Improvement: Decision Threshold Optimization

Although the model achieves approximately 99% accuracy, reducing False Negatives is an important objective for industrial quality control.

A potential next step is to optimize the classification decision threshold. Lowering the threshold for predicting the defective class (NG) may increase defect detection sensitivity and reduce False Negatives, although this can also increase False Positives.

The objective is to find an appropriate balance between defect detection performance and unnecessary product rejection, based on the quality requirements of the manufacturing process.

## Technologies

- **Programming Language:** Python
- **Deep Learning:** PyTorch, Torchvision
- **Computer Vision:** OpenCV
- **Data Processing:** NumPy
- **Machine Learning & Evaluation:** Scikit-learn
- **Visualization:** Matplotlib

## Future Improvements

- Experiment with different CNN architectures and compare their performance.
- Optimize classification thresholds to reduce False Negatives.
- Investigate model generalization on unseen industrial images.
- Explore more advanced Computer Vision approaches, such as object detection and defect localization.
- Optimize inference speed for potential industrial deployment.

