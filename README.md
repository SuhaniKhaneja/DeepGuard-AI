# 🧠 Deepfake Detection using CNN, DenseNet & EfficientNet

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 🚀 Overview
Deepfake detection is a critical problem in modern AI.  
This project compares multiple deep learning models to identify fake vs real images with high accuracy.

---

## 🏗️ Models Used
- 🧩 **Custom CNN** – Baseline model  
- 🧠 **DenseNet121** – Best performing model  
- ⚡ **EfficientNetB0** – Lightweight alternative  

---

## 📊 Results

### 🔹 Model Performance Comparison
![Performance](performance.png)

### 🔹 Confusion Matrix Comparison
![Confusion](confusion.png)

📌 **Key Result:**  
DenseNet121 achieved the highest accuracy (~93%) and showed the lowest misclassification rate.

---

## 💻 Demo / Output
![UI](ui.png)

Users can upload an image and instantly get:
- Prediction (Real / Fake)
- Confidence Score

---

## ⚙️ Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- Matplotlib  

---

## 📦 Installation
```bash
pip install -r requirements.txt
