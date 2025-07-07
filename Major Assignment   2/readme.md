
https://drive.google.com/drive/folders/1x4O_227C_L1-8RS3lgUhuJqriA5vFSFg
(google link for .pt file greater than 25mb so wasnt able to upload directly)
•	Used a CNN model to classify dog breeds 
•	Based it on MobileNetV2, which is already trained on ImageNet 
•	Froze the base model first, and trained my own classifier on top: 
•	Global Average Pooling 
•	Dropout •	Fully connected layer with softmax 
•	Tracked performance using
•	Accuracy 
•	Validation loss



test accuracy kind of limited due to long model training time when increasing number of epochs 
