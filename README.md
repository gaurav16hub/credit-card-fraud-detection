Credit Card Fraud Detection

Project Overview

This project uses Machine Learning to identify potentially fraudulent credit card transactions using Logistic Regression.

Objectives

- Detect fraudulent credit card transactions.
- Preprocess and normalize transaction data.
- Handle class imbalance using class weights.
- Evaluate the model using Precision, Recall, F1-score and Average Precision.

Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib

Dataset

Credit Card Fraud Detection dataset from Kaggle.

Dataset Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

- Class 0: Genuine transaction
- Class 1: Fraudulent transaction

Model

Logistic Regression with balanced class weights.

Results

- Accuracy: Approximately 98%
- Fraud Recall: Approximately 92%
- Average Precision Score: 0.7159

Note: Fraud precision is approximately 6%, so the model needs further improvement to reduce false fraud alerts.

Project Files

- "fraud_detection.py" — Model training and evaluation
- "fraud_model.pkl" — Trained model
- "scaler.pkl" — Data scaler
- "confusion_matrix.png" — Confusion matrix
- "precision_recall_curve.png" — Precision-recall graph

How to Run

Install the required libraries:

python -m pip install pandas numpy scikit-learn matplotlib seaborn joblib

Download "creditcard.csv" from Kaggle, place it in the project folder, and run:

python fraud_detection.py

Disclaimer

This is an educational Machine Learning project and is not intended for production financial transactions.
