### 1. Short description

This is a repository for face antispoofing detection using deep learning aproach. It contains a set of scripts that let you train a convolutional neuronal network and predict if a photo is real or spoofed.

### 2. Steps:

1. run 1_extractFaces.py - extracts the faces from test images

2. run 2_train_livenessNet.py - train the neuronal network for antispoofing detection.

3. run 3_test_livenessNet.py - predict if a photo is real or fake using the trained model.

### 3. Results

Liveness CNN Results

Confusion Matrix

|         | Spoofed | Real |
| ------- | :-----: | ---: |
| Spoofed |  9695   |  212 |
| Real    |   406   | 6009 |

|         | Precision | Recall | F1-Score | Support |
| ------- | :-------: | :----: | :------: | :-----: |
| Spoofed |   0.91    |  0.92  |   0.95   |  9907   |
| Real    |   0.91    |  0.90  |   0.92   |  6806   |

---

> Packages
>
> - tensorflow version 2.2.0
> - keras version 2.4.3
