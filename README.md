# Glaucoma Detection CNN using U-Net Segmentation

This project aims to detect glaucoma by segmenting the optic disc and optic cup in retinal images using a U-Net based convolutional neural network.

## Overview

Glaucoma is a condition that causes damage to the eye's optic nerve and can result in vision loss. Accurate segmentation of the optic disc and optic cup in retinal images is crucial for diagnosing glaucoma. This project leverages a U-Net model for image segmentation to achieve this goal.

## Features

- **Image Segmentation**: Utilizes a U-Net model to segment the optic disc and optic cup in retinal images.
- **Cup-to-Disc Ratio Calculation**: Computes the cup-to-disc ratio (CDR), a critical parameter in glaucoma diagnosis.
- **Graphical User Interface (GUI)**: Provides a user-friendly interface for loading images, performing segmentation, and displaying results.

## Requirements

- Python 3.x
- numpy
- opencv-python
- tensorflow
- tkinter
- Pillow
- matplotlib

You can install the required packages using:
```bash
pip install numpy opencv-python tensorflow Pillow matplotlib
```
