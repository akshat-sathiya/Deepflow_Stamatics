# DeepFlow Stamatics — Major Assignment 2

## Dog Breed Classification using Transfer Learning

Image classification task to identify 120 dog breeds from photographs using a fine-tuned CNN on the Kaggle Dog Breed Identification dataset.

### Dataset
- ~10,000 labeled training images across 120 breed classes
- 80/20 train-test split with stratified breed distribution
- Images resized to 224×224 and normalized to [0,1] tensors

### Approach
- Used a pretrained ResNet-18 (ImageNet weights) as the base feature extractor
- Froze the base convolutional layers and trained a custom classifier head on top:
  - Global Average Pooling
  - Dropout for regularization
  - Fully connected layer with softmax over 120 classes
- Trained for 15 epochs using CrossEntropyLoss and Adam optimizer (lr=0.001)
- Tracked training/validation accuracy and loss across epochs
- Test accuracy was limited due to long training times constraining the number of epochs

### Results
- Generated per-class softmax probabilities on the unseen test set
- Final predictions submitted as breed labels in `submission.csv`
- Training and validation loss/accuracy curves plotted to monitor overfitting

### Tech Stack
Python · PyTorch · torchvision · scikit-learn · matplotlib · Kaggle (T4 GPU)

### Files
- `cnn-model.ipynb` — Full training and inference pipeline
- `model_state_dict.pt` — Saved model weights ([Google Drive](https://drive.google.com/drive/folders/1x4O_227C_L1-8RS3lgUhuJqriA5vFSFg))
- `submission.csv` — Final predictions
