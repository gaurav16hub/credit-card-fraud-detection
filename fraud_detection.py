import joblib 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve
)
from sklearn.utils.class_weight import compute_class_weight

# 1. Load dataset
print("Loading dataset...")

df = pd.read_csv("creditcard.csv")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# 2. Check missing values and class distribution
print("\nMissing values:", df.isnull().sum().sum())
print("\nClass distribution:")
print(df["Class"].value_counts())

# 3. Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# 4. Split data before fitting the scaler
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Normalize Time and Amount
scaler = StandardScaler()

X_train = X_train.copy()
X_test = X_test.copy()

X_train[["Time", "Amount"]] = scaler.fit_transform(
    X_train[["Time", "Amount"]]
)

X_test[["Time", "Amount"]] = scaler.transform(
    X_test[["Time", "Amount"]]
)

# 6. Handle class imbalance using class weights
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights = dict(zip(classes, weights))

# 7. Train the model
print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    class_weight=class_weights,
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed!")

# 8. Predict on unseen test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 9. Evaluate model
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Genuine", "Fraud"],
    zero_division=0
))

print("Average Precision Score:",
      round(average_precision_score(y_test, y_prob), 4))

# 10. Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Genuine", "Fraud"]
)
disp.plot(cmap="Blues", values_format="d")
plt.title("Credit Card Fraud Detection")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# 11. Plot precision-recall curve
precision, recall, thresholds = precision_recall_curve(
    y_test, y_prob
)

plt.figure(figsize=(8, 5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("precision_recall_curve.png", dpi=150)
plt.show()

print("\nProject completed successfully!")
# Save trained model and scaler
joblib.dump(model, "fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully!")
print("Scaler saved successfully!")