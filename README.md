# ENCS5343-Computer-Vision-Arabic-Handwritten-Text-Writer-Identification

# Arabic Handwritten Text Writer Identification (AHAWP) Project

This repository contains the code and experiments conducted for the **Arabic Handwritten Automatic Word Processing (AHAWP)** project. The goal is to **classify handwritten Arabic words** by identifying the writer (82 classes). We explore several deep learning strategies ranging from **custom CNNs** to **pretrained ResNet architectures**, leveraging **data augmentation** and **transfer learning** to achieve high accuracy on the AHAWP dataset.

---


## Overview
Handwritten text recognition is challenging due to variations in writing styles, stroke pressures, and connected letter shapes (especially in Arabic). In this project:
- **Task 1** focuses on building and tuning **custom CNNs**.
- **Task 2** introduces **data augmentation** to improve generalization.
- **Task 3** compares a **well-known CNN architecture (ResNet-18)** to the custom models.
- **Task 4** explores **transfer learning** with **ResNet-50** pretrained on ImageNet to push accuracy even higher.

By the end, we demonstrate that combining data augmentation with transfer learning can significantly enhance handwriting recognition performance.

---

## Dataset
The AHAWP dataset used in this project can be found on [Kaggle](https://www.kaggle.com/datasets/m7mdodeh/isolated-words-per-user/). It consists of:
- **82 writers**  
- **10 Arabic words** (each writer produces 10 samples per word)  
- **8,144 total images** (grayscale PNGs)

You will need to download this dataset and follow the instructions below to integrate it with the code in this repository.

---

## Tasks Breakdown
1. **Task 1: Custom CNN Architectures**  
   - Compare three CNN designs (varying depth and convolutional filters).  
   - Tune hyperparameters (learning rate, batch size, epochs, etc.).  

2. **Task 2: Data Augmentation**  
   - Apply offline augmentation (rotation, shear, scaling, translation).  
   - Retrain the best model from Task 1 on the augmented dataset.  

3. **Task 3: Well-Known CNN (ResNet-18)**  
   - Adapt ResNet-18 for grayscale input.  
   - Train from scratch on the augmented dataset.  

4. **Task 4: Transfer Learning (ResNet-50)**  
   - Use a ResNet-50 pretrained on ImageNet.  
   - Partially fine-tune higher layers for Arabic handwriting.  

---

## Installation & Requirements
- **Python 3.7+**
- **PyTorch 1.7+**
- **Torchvision 0.8+**
- **NumPy 1.19+**
- **Matplotlib**
- **Pillow**
- **OpenCV-Python**  
- (Optional) **CUDA**-enabled GPU for faster training


## Results Summary
| **Model / Task**                     | **Final Test Accuracy** | **Training Time (sec)** |
|--------------------------------------|--------------------------|--------------------------|
| Custom CNN (Task 1, No Aug)          | ~65.07%                 | ~218                    |
| Custom CNN + Aug (Task 2)            | ~76.55%                 | ~611                    |
| ResNet-18 + Aug (Task 3)             | ~90.12%                 | ~1,208                  |
| ResNet-50 (Pretrained) + Aug (Task 4)| ~98.77%                 | ~2,292                  |


## License
This project is provided under the MIT License. Feel free to use, modify, and distribute the code with appropriate attribution.
