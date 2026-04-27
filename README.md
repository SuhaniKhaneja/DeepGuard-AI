# 🧠 Deepfake Detection using CNN, DenseNet & EfficientNet

## 🚀 Overview

This project focuses on detecting deepfake images using deep learning techniques. Multiple models were implemented and compared to determine the most effective approach.

---

## 🏗️ Models Used

* Custom CNN (Baseline)
* DenseNet121 (Best Performance)
* EfficientNetB0 (Efficient Alternative)

---

## 📊 Model Comparison

![Model Comparison](assets/model_comparison.png)

DenseNet121 outperformed other models across all evaluation metrics.

---

## 📈 Training Performance

### DenseNet121

![DenseNet Training](assets/densenet_training.png)

### EfficientNetB0

![EfficientNet Training](assets/efficientnet_training.png)

---

## 📊 Evaluation Results

### DenseNet121 Results

![DenseNet Results](assets/densenet_results.png)

### EfficientNetB0 Results

![EfficientNet Results](assets/efficientnet_results.png)

---

## 🔬 Custom CNN (Baseline)

A custom CNN was implemented as a baseline model. It showed signs of overfitting, highlighting the advantage of transfer learning approaches.

![CNN Training](assets/cnn_training.png)

---

## 🖥️ Web Interface

![UI](assets/ui_prediction.png)

A user interface was developed to demonstrate deepfake detection. Users can upload an image and receive predictions with confidence scores.

---

## ⚙️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* Matplotlib

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 📌 Key Insights

* DenseNet121 achieved highest accuracy (~93%)
* Transfer learning outperformed custom CNN
* Model comparison helped identify best architecture

---

## 🎯 Future Work

* Add real-time deployment
* Extend to video deepfake detection

---

## 👩‍💻 Author

Suhani Khaneja
