# AI Tools and Frameworks Assignment 🚀

## Overview
This repository contains my AI Tools and Applications assignment titled **"Mastering the AI Toolkit"**.  
The project demonstrates practical applications of AI frameworks, including classical machine learning, deep learning, and natural language processing, along with ethical AI practices and deployment.

---

---

## Tools & Frameworks Used
- **TensorFlow 2.x** – CNN on MNIST dataset  
- **Scikit-learn** – Decision Tree on Iris dataset  
- **spaCy** – Named Entity Recognition & sentiment analysis on Amazon reviews  
- **Streamlit** – Web interface for MNIST classifier  
- **Jupyter Notebook** – Interactive notebooks for analysis & visualization  

---

## 1. Iris Decision Tree Analysis
- **Dataset:** Iris species dataset  
- **Goal:** Predict iris species based on sepal and petal dimensions  
- **Steps:**
  1. Preprocessing (handle missing values, encode labels)  
  2. Train a **Decision Tree Classifier**  
  3. Evaluate using **accuracy, precision, and recall**  
- **Results:**  
  - High accuracy (~98–99%) in predicting species  
  - Visualized tree for understanding feature importance  
- **Notes:** Demonstrates classical machine learning with scikit-learn.

---

## 2. MNIST CNN Classifier
- **Dataset:** MNIST handwritten digits  
- **Goal:** Build a **Convolutional Neural Network** (CNN) to classify digits  
- **Steps:**
  1. Preprocess images (normalize, add channel dimension)  
  2. Define CNN architecture (Conv2D → MaxPooling → Dense layers)  
  3. Train model with **5 epochs**  
  4. Evaluate model performance on test set  
  5. Visualize predictions on 5 sample images  
- **Results:**
  - Test accuracy: **99.07%**  
  - Loss: 0.0324  
  - Model can confidently classify handwritten digits  
- **Explanation:** Chose TensorFlow over PyTorch for simplicity, smaller dataset, and beginner-friendly deep learning implementation.

---

## 3. Amazon Review Analysis with spaCy
- **Dataset:** Amazon product reviews  
- **Goal:** Extract **named entities** (brands, products) and perform **sentiment analysis**  
- **Steps:**
  1. Load reviews in Python  
  2. Use **spaCy NER** to extract product names & brands  
  3. Apply **rule-based sentiment analysis** (positive/negative)  
- **Results:**  
  - Successfully extracted key product mentions and brands  
  - Simple sentiment classification for each review  
- **Notes:** Demonstrates NLP capabilities and text analysis beyond basic string operations.

---

## 4. Ethical Considerations
- **Potential Biases:**
  - MNIST dataset: predominantly digits written by certain demographics  
  - Amazon reviews: skewed by user language or rating distribution  
- **Mitigation Strategies:**
  - **TensorFlow Fairness Indicators** for detecting model bias  
  - **spaCy rule-based checks** to ensure consistent entity extraction  
- **Importance:** Ensures responsible AI practices and fairness in predictions.

---

## 5. Streamlit Deployment
- **Goal:** Deploy the MNIST CNN classifier as a web app for easy interaction  
- **How it works:**  
  1. Upload or draw a digit  
  2. App predicts the digit using the trained CNN  
- **Requirements:** `requirements.txt` includes all necessary packages  
- **Live Demo Link:**  
[🌐 Try the MNIST Classifier Online](https://ai-tools-and-frameworks-tmx9tan3svmgwggpbdkpnk.streamlit.app/)

---
By Moraa Robert 
Email: **moraarobert20@gmail.com**
# AI-tools-and-frameworks
