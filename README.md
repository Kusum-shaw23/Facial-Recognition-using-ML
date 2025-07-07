# 👤 Face Recognition using Machine Learning

This project implements a **face recognition system** using machine learning techniques. It is trained on a publicly available dataset from Kaggle and supports prediction and identification of faces in images. The goal is to build a robust and efficient model that can recognize faces with high accuracy.

## 📁 Dataset

- **Source:** [Kaggle Face Dataset](https://www.kaggle.com/)  
- **Description:** The dataset contains labeled facial images used for training and testing. Each class corresponds to a unique individual.

> **Note**: Due to Kaggle’s terms of use, please download the dataset manually and place it in the `/dataset` directory.

## 🧠 Model & Techniques Used

- Face detection using `OpenCV` or `dlib`
- Feature extraction using:
  - PCA (Principal Component Analysis) or CNN embeddings
- Classification using:
  - Support Vector Machines (SVM)
  - K-Nearest Neighbors (KNN)
  - or a CNN 

## 📦 Dependencies

Install the required Python libraries using:

```bash
pip install -r requirements.txt
