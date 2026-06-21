# DeepFlow Stamatics — Major Assignments

Machine learning projects completed as part of the DeepFlow course by Stamatics, covering regression on structured data and image classification with deep learning.

## Assignment 1 — Bulldozer Sale Price Prediction
Built a regression pipeline to predict heavy equipment auction prices from tabular data (~30k samples, 80+ raw features). Reduced feature space to 16 columns by filtering high-null features, imputed missing values with median, and encoded categoricals using pandas category codes. Applied log transform on the target variable to align with the RMSLE evaluation metric, then trained a Random Forest Regressor for final predictions.

**Tech:** Python · pandas · NumPy · scikit-learn

## Assignment 2 — Dog Breed Classification
Developed an image classification model to identify 120 dog breeds using transfer learning. Used a pretrained ResNet-18 as a frozen feature extractor and added a custom classifier head (global average pooling → dropout → fully connected + softmax). Trained for 15 epochs with CrossEntropyLoss and Adam, monitoring train/validation loss and accuracy curves to check for overfitting. Generated per-class softmax probabilities on the test set for final submission.

**Tech:** Python · PyTorch · torchvision · scikit-learn · Kaggle GPU
