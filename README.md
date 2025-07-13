# 🧠 Facial Expression-Based Lie Detection Using Deep Learning

This project aims to classify facial expressions into two categories — **Truth** and **Lie** — using deep learning. It leverages **transfer learning** with **MobileNetV2** for robust  performance on a labeled Kaggle dataset of facial expressions.

---

## 📁 Dataset Overview

The dataset is sourced from [Kaggle](https://www.kaggle.com), containing labeled facial expression images.

* **Classes:**

  * `truth`
  * `lie`
* **Training Images:** 14,108
* **Validation Images:** 3,526

---

## 🧠 Model Architecture

We use **MobileNetV2** as a frozen base model and build a custom classification head:

```
Input Image (128x128x3)
↓
MobileNetV2 (pretrained on ImageNet, frozen)
↓
GlobalAveragePooling2D
↓
BatchNormalization
↓
Dense (64 units, ReLU)
↓
Dropout (0.5)
↓
Dense (1 unit, Sigmoid)
↓
Output: Truth / Lie
```

---

## ⚙️ Training Configuration

* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam
* **Metric:** Accuracy
* **Epochs:** 20
* **Batch Size:** 32
* **Image Size:** 128×128
* **Augmentation:** Horizontal flip, rotation, zoom

---

## 📊 Training Results

* **Training Accuracy:** \~65%
* **Validation Accuracy:** \~45%
* **Training Accuracy:** \Peaked around 74%
* **Validation Accuracy:** \Fluctuated between 40%–60%
* **Training Loss:** \Decreased to ~0.45
* **Validation Loss:** \Ranged from 0.77 to 0.95
 

> 🔍 *Note:* The model shows signs of overfitting. Further improvements can be made through fine-tuning, better regularization, and more diverse data.

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/facial-lie-detection
   cd facial-lie-detection
   ```

2. Download the dataset from Kaggle and place images in:

   ```
   data/train/
   data/val/
   ```

3. Install dependencies:

   ```bash
   pip install tensorflow keras opencv-python matplotlib scikit-learn
   ```

4. Train the model:

   ```bash
   python train_model.py
   ```

---

## 🧠 Future Improvements

* Unfreeze and fine-tune MobileNetV2 layers
* Try advanced models like EfficientNet or custom CNNs
* Apply better class balancing techniques
* Use temporal features for video-based lie detection

---

## 📁 Project Structure

```
📦 facial-lie-detection/
├── data/
│   ├── train/
│   └── val/
├── train_model.py
├── model.h5
└── README.md
```

---
