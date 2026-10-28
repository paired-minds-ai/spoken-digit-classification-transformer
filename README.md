# Spoken Digit Classification with Transformers

A deep learning project exploring spoken digit recognition using a combination of 1D convolutional layers and Transformer encoders, with additional experiments investigating robustness to noise.

## Project Overview

The goal of this project is to classify spoken digits (0–9) from audio-derived spectrogram features using deep learning.

Two related experiments are included:

### 1. Baseline Spoken Digit Classification

Implementation of a Conv1D–Transformer architecture for spoken digit recognition, including feature extraction, positional encoding, Transformer-based sequence modeling, and multiclass classification.

### 2. Noise-Augmented Training

An extension of the baseline experiment investigating whether injecting noise into training data can improve the model's robustness when classifying noisy audio.


## Key Techniques

- Spectrogram-based audio representation
- Conv1D feature extraction
- Positional encoding
- Transformer encoder with multi-head self-attention
- Data augmentation
- Multiclass classification
- Confusion matrix and performance evaluation
- Evaluation under different signal-to-noise ratios (SNRs)

## Dataset

The experiments use the Free Spoken Digit Dataset (FSDD), containing recordings of spoken digits from 0 to 9.

Dataset: [Free Spoken Digit Dataset on GitHub](https://github.com/Jakobovski/free-spoken-digit-dataset)

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Repository Structure

```text
spoken-digit-classification-transformer/
├── README.md
├── spoken_digit_transformer.ipynb
└── spoken_digit_transformer_noise_augmented.ipynb
```

## Results

Quantitative results for both experiments will be documented after evaluation, including classification performance and robustness under noisy conditions.

## Limitations

Performance may depend on the dataset split, preprocessing choices, augmentation strategy, and noise conditions. Comparisons between experiments should use consistent evaluation data and procedures.

## Acknowledgements

This project was developed as part of undergraduate coursework in deep learning.
