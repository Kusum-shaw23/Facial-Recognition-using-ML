👤 Face Recognition using Machine Learning

A machine learning–based system that detects and recognizes faces from images using feature extraction and classification techniques. Trained on a Kaggle facial dataset, the model achieves high accuracy in identifying individuals.

📁 Dataset

Source: Kaggle Face Dataset

Description: Labeled facial images, each representing a unique individual.

Note: Download the dataset manually from Kaggle and place it in the /dataset directory (per Kaggle’s terms of use).

🧠 Model & Techniques

Face Detection: OpenCV or dlib

Feature Extraction:

PCA (Principal Component Analysis)

CNN embeddings

Classification:

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Convolutional Neural Network (CNN)

📦 Dependencies

Install Python dependencies:

pip install -r requirements.txt

🚀 Quick Start
# 1. Clone repo
git clone https://github.com/your-username/face-recognition-ml.git
cd face-recognition-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add dataset
# Place Kaggle dataset in /dataset folder

# 4. Run training
python train.py

# 5. Test model
python predict.py --image path/to/image.jpg

📈 Future Enhancements

Integrate deep learning–based face recognition (e.g., FaceNet)

Add real-time recognition from webcam

Deploy as a web application

Author: Kusum Shaw
License: MIT
