EEG Classification Model for Epileptic Seizure Detection
**Project Overview**

This project implements an end-to-end machine learning and deep learning pipeline for automated epileptic seizure detection using EEG (Electroencephalography) signals. The goal is to analyze raw EEG recordings, extract meaningful patterns, and accurately classify seizure versus non-seizure activity.

The project compares classical machine learning models with modern deep learning architectures to understand their strengths and limitations when applied to real-world biomedical signal data.

**Motivation**

Manual interpretation of EEG signals is time-consuming, requires clinical expertise, and is prone to human error—especially in long, multi-channel recordings. Automated seizure detection systems can assist clinicians by improving accuracy, reducing workload, and enabling faster diagnosis.

This project explores how data preprocessing, feature engineering, and model selection impact seizure detection performance across datasets with different noise levels and complexity.

**Datasets Used**

Two publicly available EEG datasets were used to ensure robust evaluation:

**1. CHB-MIT Scalp EEG Dataset**

- Multi-channel EEG recordings (23 channels)

- Long clinical recordings (~1 hour per subject)

- Seizure events annotated by medical experts

- Highly imbalanced and noisy, closely resembling real hospital data

**2. Bonn EEG Dataset**

- Single-channel EEG recordings

- Pre-segmented and labeled signals

- Five subsets (Z, O, N, F, S) representing different neurological states

- Clean and well-structured, ideal for benchmarking models

**Data Preprocessing**

Raw EEG signals were processed through a structured pipeline to ensure model-ready input:

- Band-pass filtering (0.5–40 Hz) to retain meaningful brain activity

- Notch filtering to remove power-line noise

- Resampling to reduce computational complexity

- Sliding window segmentation with overlap (CHB-MIT)

- Automatic seizure labeling based on annotation overlap

- Z-score normalization applied per channel

This preprocessing ensures consistency across datasets while preserving important temporal patterns.

**Feature Extraction**

For classical machine learning models, handcrafted features were extracted from EEG signals, including:

- Statistical features (mean, standard deviation, skewness, kurtosis)

- Signal characteristics (energy, peak-to-peak amplitude)

- Frequency-domain features using Welch’s Power Spectral Density

- EEG band powers (delta, theta, alpha, beta, gamma)

Each EEG segment was represented as a 13-dimensional feature vector.

**Models Implemented**
Random Forest (Classical ML)

- Trained on extracted feature vectors

- Class-weighted to handle seizure imbalance

- Performs well on clean data but struggles with noisy clinical EEG

**Convolutional Neural Network (CNN)**
- Trained directly on raw EEG windows

- Learns temporal and spatial signal patterns automatically

- Achieves the best seizure detection performance

- Demonstrates strong generalization on complex EEG data

**Long Short-Term Memory (LSTM)**

- Sequential deep learning model

- Designed for temporal dependencies

- Exhibits overfitting and lower seizure recall due to class imbalance and long 

**Model Evaluation**

Models were evaluated using clinically relevant metrics:

- Accuracy

- Precision

- Recall

- F1-score

- Confusion matrices

- Training and validation loss curves

**Key Findings**

- CNN significantly outperforms classical models on multi-channel clinical EEG

- Random Forest performs well only on clean, structured data

- LSTM shows instability and poor seizure recall without additional preprocessing

**Results Summary**

- CNN achieved the highest seizure recall and overall accuracy on the CHB-MIT dataset

- Feature-based models fail to capture complex temporal EEG patterns

- Deep learning models are better suited for real-world biomedical signal analysis

**Technologies Used**
- Python

- NumPy, SciPy, Pandas

- Scikit-learn

- TensorFlow / Keras

- Matplotlib / Seaborn

- EEG signal processing techniques

**Key Takeaways**

- Real-world EEG data is noisy and highly imbalanced

- Proper preprocessing is critical for model performance

- CNNs excel at learning raw signal representations

- Classical ML models are limited when dealing with complex temporal data

**Future Improvements**

- Spectrogram-based 2D CNN architectures

- Transformer-based EEG models

- Synthetic data augmentation for rare seizure events

- Real-time seizure detection deployment

- Advanced feature engineering techniques

