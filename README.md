# DEAP Emotion Classification with CNNs

This repository implements emotion classification on the DEAP dataset using multiple CNN-based architectures. The models operate on different levels of feature complexity: **1D CNN**, **2D CNN**, and **3D CNN**.

---

##  Repository Structure
├── CNN_1D.py # Train 1D CNN model
├── CNN_2D.py # Train 2D CNN model
├── CNN_3D.py # Train 3D CNN model
├── DeapDataset.py # Custom PyTorch dataset with optional 1D/2D/3D loading
├── feature_extraction.py # Extract features from raw DEAP data
├── models.py # CNN model definitions
├── pipeline.py # Training pipeline class (training + evaluation + plotting)
└── README.md # This file
