# DEAP Emotion Recognition using CNNs

This repository contains code for training 1D, 2D, and 3D Convolutional Neural Networks (CNNs) for emotion recognition using the DEAP dataset.

##  File Structure

```
├── CNN_1D.py              # Train 1D CNN model
├── CNN_2D.py              # Train 2D CNN model
├── CNN_3D.py              # Train 3D CNN model
├── DeapDataset.py         # Custom PyTorch dataset with optional 1D/2D/3D loading
├── feature_extraction.py  # Extract features from raw DEAP data
├── models.py              # CNN model definitions
├── pipeline.py            # Training pipeline class (training + evaluation + plotting)
├── train_data.npy         # Preprocessed training features
├── train_label.npy        # One-hot encoded training labels
├── test_data.npy          # Preprocessed testing features
├── test_label.npy         # One-hot encoded testing labels
└── README.md              # This file
```


##  Usage


```bash
# Train a 1D CNN model
python CNN_1D.py

# Train a 2D CNN model
python CNN_2D.py

# Train a 3D CNN model
python CNN_3D.py
```
