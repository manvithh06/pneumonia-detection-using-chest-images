# Pneumonia Detection from Chest X-Rays

A deep learning project that detects pneumonia from chest X-ray images. It compares two approaches — a Convolutional Neural Network (CNN) trained from scratch, and a transfer-learning model built on top of a pretrained VGG16 network — and evaluates which one performs better at this medical imaging task.

## Problem Statement

Pneumonia is diagnosed by examining chest X-rays for signs of infection in the lungs. Manually reading X-rays is time-consuming and depends on the expertise of the radiologist. This project builds an automated binary classifier that takes a chest X-ray image and predicts whether it shows a **NORMAL** lung or **PNEUMONIA**.

## Dataset

- **Source:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset on Kaggle
- **Size:** ~5,856 X-ray images, split into `train`, `val`, and `test` folders, each containing `NORMAL` and `PNEUMONIA` subfolders
- **Class distribution:** The dataset is imbalanced — significantly more PNEUMONIA images than NORMAL, which is handled using class weighting during training
- **Image size used:** All images resized to 224×224 pixels, RGB

## Approach

### 1. Baseline CNN (trained from scratch)
A custom Convolutional Neural Network built with 4 convolutional blocks (32 → 64 → 128 → 256 filters), each followed by max pooling, then a dense classification head with dropout for regularization. This model learns visual features entirely from the X-ray dataset itself, with no external knowledge.

### 2. Transfer Learning (VGG16)
A VGG16 network pretrained on ImageNet is used as a feature extractor. Its convolutional base is initially frozen and only a custom classification head is trained on top of it. The model is then fine-tuned by unfreezing the last convolutional block (`block5`) and continuing training at a very low learning rate, allowing the network to slightly adapt its high-level features to X-ray images specifically.

Data augmentation (random flips, rotation, zoom, contrast) is applied during training to reduce overfitting given the relatively small dataset size.

## Evaluation

Both models are evaluated on a held-out test set using:
- Accuracy, Precision, Recall, F1-score
- ROC-AUC and ROC curve comparison
- Confusion matrix
- Precision-recall curve with F1-optimized threshold tuning

Recall is treated as a particularly important metric here, since missing a true pneumonia case (a false negative) is more costly than a false alarm.

## Model Interpretability

Grad-CAM (Gradient-weighted Class Activation Mapping) is used on the transfer-learning model to visualize which regions of the X-ray the model focuses on when making its prediction. This helps verify that the model is attending to relevant lung regions rather than irrelevant image artifacts.

## Tech Stack

- **Language:** Python
- **Framework:** TensorFlow / Keras
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, OpenCV
- **Environment:** Google Colab (GPU-accelerated training)

## Project Structure

```
├── Pneumonia_detection_from_chest_xrays.ipynb   # Main notebook: data loading, EDA,
│                                                 # baseline CNN, transfer learning,
│                                                 # evaluation, Grad-CAM
├── pneumonia_baseline_cnn.keras                 # Saved baseline CNN model
├── pneumonia_vgg16_transfer.keras               # Saved transfer-learning model
└── README.md
```

## How to Run

1. Download the dataset from Kaggle (`paultimothymooney/chest-xray-pneumonia`)
2. Open the notebook in Google Colab (GPU runtime recommended)
3. Upload your `kaggle.json` API key when prompted, to download the dataset directly
4. Run the notebook cells in order — data loading → baseline CNN → transfer learning → evaluation → Grad-CAM

## Results Summary

Both models are compared side-by-side on accuracy, precision, recall, F1-score, and ROC-AUC on the test set, with the transfer-learning model generally achieving stronger and more stable performance due to the visual features it inherits from pretraining on ImageNet.

## Future Improvements

- Try additional pretrained backbones (e.g., DenseNet121 / CheXNet-style architecture)
- Expand the validation set (the original dataset's validation split is very small)
- Deploy the model as a simple web app for interactive predictions
- Experiment with ensembling the baseline CNN and transfer-learning model

## Disclaimer

This project is for educational and research purposes only. It is not a certified diagnostic tool and should not be used for actual medical decision-making.
