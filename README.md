# Chest X-Ray Classification with MobileNetV2 (Transfer Learning)

This project applies deep learning (CNN transfer learning) to classify chest X-ray images into **COVID-19, Normal**, and **Viral Pneumonia**. The goal is to explore the feasibility of an automated clinical triage support approach using radiographic patterns and to analyze clinically relevant errors via sensitivity/specificity and confusion matrices.

## Dataset

COVID-19 Chest X-Ray Database (Qatar University & University of Dhaka collaborators).
Multisource public CXR collection including COVID-19, Normal and Viral Pneumonia images.
License: **CC BY 4.0**

## Objective

To build a baseline deep learning model for **multiclass chest X-ray classification** that could serve as a **triage support tool**, and to assess performance using clinically meaningful metrics (confusion matrix, sensitivity/specificity per class, ROC-AUC).

## Methods
### Modeling approach

Multiclass image classification using **Convolutional Neural Networks (CNN) with transfer learning**.

Pre-trained **MobileNetV2** architecture (ImageNet weights) used as a feature extractor.

Final classification head trained for three classes: COVID-19, Normal, Viral Pneumonia.

### Image preprocessing

Original chest X-ray images are provided at 256×256 pixels resolution.

All images were resized to **224×224** pixels to match the canonical input size of the pre-trained MobileNetV2 architecture and ensure efficient transfer learning.

Pixel values were normalized to the [0,1]range.

### Data splitting

Stratified **80/20 train–validation split performed per class**, preserving class proportions.

### Training strategy
The convolutional backbone was initially frozen to preserve learned low-level visual features.

Only the classification head was trained.

Optimization performed using Adam optimizer with early stopping based on validation loss.

### Data augmentation
Light, anatomy-preserving augmentations were applied during training:
-Small rotations
-Minor translations and zoom
No aggressive geometric or contrast transformations were applied to prevent the model from learning non-clinical artifacts.

### Evaluation metrics
-Overall accuracy
-**Class-wise precision, recall (sensitivity)** and **F1-score**
-**Confusion matrix**
-Multiclass **ROC-AUC (One-vs-Rest)** with macro and weighted averaging

## Key Results
-Overall validation accuracy: ~0.90
-Macro ROC-AUC (OvR): **~0.98**, indicating strong overall discriminative ability.
-High specificity for pathological classes, particularly **COVID-19**.

### Class-wise performance highlights
-**COVID-19** : Specificity: **0.99**, Sensitivity (recall): **0.67**
-**Normal** : High sensitivity, slightly reduced precision
-**Viral Pneumonia** : Balanced sensitivity and specificity

## Clinical Interpretation and Triage-Oriented Perspective

The model demonstrates **very high specificity for COVID-19 (≈0.99)**.
This indicates that when the model predicts COVID-19, it does so with high confidence and produces **very few false positive COVID classifications**. In practical terms, the model is highly reliable at recognizing **pathological COVID-related patterns** when they are present.

Conversely, **COVID-19 sensitivity is moderate (≈0.67)**. Analysis of the confusion matrix shows that a subset of true COVID-19 cases are misclassified as Normal. These represent **false negative COVID cases**, not false positives.

Importantly, this behavior is **clinically informative rather than purely erroneous**.

### Triage implications
From a triage perspective, the model’s behavior suggests the following workflow interpretation:

- **Cases predicted as COVID-19** can be considered **high-confidence pathological studies**, suitable for fast-tracking further evaluation, isolation protocols, or confirmatory testing.

- **Cases predicted as Normal**, particularly in high-prevalence or high-risk contexts, should not be automatically dismissed. The observed false negatives indicate that this group may benefit from **secondary review, clinical correlation, or follow-up imaging**.

Thus, the model effectively **facilitates prioritization of clinical processes**, rather than replacing medical judgment.

## Clinical trade-off and future optimization

The model is intentionally optimized for **high specificity in COVID-19 detection**, minimizing false positive diagnoses. This conservative behavior is well suited for **clinical decision-support and triage workflows**, where avoiding unnecessary escalation of care and resource utilization is critical.

Future work will focus on **increasing COVID-19 sensitivity** through cost-sensitive learning and targeted fine-tuning of deeper convolutional layers, while preserving the model’s strong capacity to identify pathological lung patterns across the entire field of view.

## Project Structure
```text
.
├── data/
│   └── raw/
│       └── README_dataset.md       # Original dataset description and references
│
├── models/
│   └── xray_cnn_mobilenetv2.keras  # Trained MobileNetV2-based CNN model
│
├── results/
│   ├── classification_report.txt  # Precision, recall, F1-score per class
│   ├── roc_auc.txt                # Macro and weighted ROC-AUC scores
│   ├── confusion_matrix.png       # Confusion matrix (validation set)
│   ├── accuracy_curve.png         # Training vs validation accuracy
│   ├── loss_curve.png             # Training vs validation loss
│   └── recall_per_class.png       # Sensitivity (recall) per class
│
├── src/
|   └── train_mobilenetv2.py       # End-to-end training and evaluation script
├── .gitignore                     # Git ignore rules (e.g. virtual environment)
└── README.md                      # Project documentation
```

#### Note:
Raw image data are not included in this repository.
The folder structure is preserved to ensure reproducibility, while images must be obtained from the original public dataset (see Dataset section).

## Technical Stack
- **Programming language:** Python  
- **Deep Learning:** Convolutional Neural Networks (CNN)  
- **Architecture:** MobileNetV2 (transfer learning, ImageNet pretrained)  
- **Framework:** TensorFlow / Keras  
- **Data pipeline:** `tf.data` API  
- **Evaluation metrics:**  
  - Accuracy  
  - Precision, Recall (Sensitivity), F1-score  
  - Specificity (derived from the confusion matrix)  
  - ROC-AUC (One-vs-Rest, macro and weighted)  
- **Visualization:** Matplotlib  
- **Environment:** Linux (Ubuntu / WSL)

## How to run

1. Clone the repository
2. Create and activate a Python virtual environment
3. Install dependencies
4. Run the training script


## Why this project matters

This project illustrates how deep learning models can be applied to real-world medical imaging workflows, emphasizing not only predictive performance but also clinically meaningful trade-offs such as sensitivity, specificity, and triage prioritization. The focus on error analysis and clinical interpretation reflects the requirements of healthcare AI systems beyond purely technical accuracy.
