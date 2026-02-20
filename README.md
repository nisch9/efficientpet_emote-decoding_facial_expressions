# 🐾 EfficientPet --- Emote Decoding using Deep Learning

An end-to-end deep learning system that detects **pet emotions from
facial expressions** using Convolutional Neural Networks (CNNs) and
deploys predictions through an interactive Streamlit web application.

------------------------------------------------------------------------

## 🚀 Project Overview

EfficientPet is designed to automatically analyze pet facial images and
classify their emotional state into predefined categories. The system
uses deep learning to extract visual features and provide real-time
predictions via a user-friendly interface.

The objective is to help pet owners, veterinarians, and researchers
better understand animal emotions in a **non-invasive and objective**
manner.

------------------------------------------------------------------------

## 🎯 Problem Statement

Understanding pet emotions is challenging due to reliance on subjective
human interpretation. This project aims to:

-   Automate pet emotion recognition\
-   Provide real-time emotion monitoring\
-   Reduce human bias in behavioral assessment\
-   Enable scalable emotion analysis systems

------------------------------------------------------------------------

## 🏗️ Technical Architecture

    User Upload → Streamlit UI → CNN/EfficientNet Model → Prediction → Display Result

------------------------------------------------------------------------

## 📂 Dataset

-   **Source:** Kaggle --- Pets Facial Expression Dataset\
-   **Format:** JPG images\
-   **Classes:** Angry, Happy, Sad, Other\
-   **Images per class:** \~250

🔗 Dataset:\
https://www.kaggle.com/datasets/anshtanwar/pets-facial-expression-dataset

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   Python\
-   TensorFlow / Keras\
-   EfficientNet\
-   OpenCV / PIL\
-   Streamlit\
-   NumPy, Pandas, Matplotlib

------------------------------------------------------------------------

# ⚙️ Implementation Guide

------------------------------------------------------------------------

## 1️⃣ Environment Setup

### Prerequisites

-   Python 3.8+\
-   VS Code (recommended)\
-   GPU (optional but recommended)\
-   Streamlit

### Install Dependencies

``` bash
pip install tensorflow streamlit numpy pandas matplotlib pillow
```

------------------------------------------------------------------------

## 2️⃣ Project Structure

    EfficientPet_Emote-Decoding_Facial_Expressions/
    │
    ├── Dataset/
    ├── Model/
    ├── Training/
    ├── web.py
    └── README.md

------------------------------------------------------------------------

## 3️⃣ Data Collection & Preparation

### Steps

1.  Download dataset from Kaggle\
2.  Load images into notebook\
3.  Verify class distribution\
4.  Configure GPU in Kaggle (optional)\
5.  Import required libraries

------------------------------------------------------------------------

## 4️⃣ Exploratory Data Analysis (EDA)

### Performed Analysis

-   Extracted class names\
-   Verified image counts\
-   Visualized sample images\
-   Checked for null values

------------------------------------------------------------------------

## 5️⃣ Data Splitting

Dataset split ratio:

    Train : Test : Validation = 8 : 1 : 1

------------------------------------------------------------------------

## 6️⃣ Image Data Generation

Used `ImageDataGenerator` for:

-   Rescaling\
-   Augmentation\
-   Batch generation\
-   Flow from directory

------------------------------------------------------------------------

## 7️⃣ Model Building

### Architecture

-   Sequential model\
-   EfficientNet backbone\
-   Dense classification head\
-   Dropout regularization\
-   Batch normalization

------------------------------------------------------------------------

## 8️⃣ Model Training

### Key Steps

-   Compile model\
-   Train on training set\
-   Validate on validation set\
-   Monitor accuracy and loss\
-   Apply callbacks

------------------------------------------------------------------------

## 9️⃣ Model Evaluation

Model performance is visualized using:

-   Training vs Validation Loss\
-   Training vs Validation Accuracy\
-   Classification report\
-   Confusion matrix

------------------------------------------------------------------------

## 🔟 Model Saving

Best model weights are saved as:

``` python
model.save_weights("EfficientPet.h5")
```

------------------------------------------------------------------------

# 🌐 Web Application (Streamlit)

The trained model is integrated into a Streamlit app for real-time
predictions.

------------------------------------------------------------------------

## 1️⃣ Install Streamlit

``` bash
pip install streamlit
```

------------------------------------------------------------------------

## 2️⃣ web.py (Core App)

### Features

-   Upload pet image\
-   Preprocess input\
-   Run model inference\
-   Display predicted emotion\
-   Show confidence score

------------------------------------------------------------------------

## 3️⃣ Run the Application

``` bash
streamlit run web.py
```

------------------------------------------------------------------------

## 🖥️ Application Flow

1.  User uploads pet image\
2.  Image is resized and normalized\
3.  Model predicts emotion\
4.  Predicted label + confidence displayed\
5.  UI updates in real time

------------------------------------------------------------------------

## 📊 Model Classes

-   Angry\
-   Happy\
-   Sad\
-   Other

------------------------------------------------------------------------

## 🧪 How to Reproduce

``` bash
# 1. Download dataset
# 2. Train model (Training/)
# 3. Save weights to Model/
# 4. Install dependencies
# 5. Run Streamlit app
streamlit run web.py
```
