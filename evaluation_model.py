import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load model and encoder
model = load_model("bim_hand_sign_model.keras")
label_encoder = joblib.load("label_encoder.pkl")

# Load dataset
df = pd.read_csv("hand_landmarks_all.csv")
# Drop BOTH 'label' and 'category' columns
X = df.drop(["label", "category"], axis=1, errors='ignore')
y = df["label"]

# Encode labels
y_encoded = label_encoder.transform(y)

# Predict
y_pred = model.predict(X)
y_pred_classes = np.argmax(y_pred, axis=1)

# Evaluation
print(classification_report(y_encoded, y_pred_classes, target_names=label_encoder.classes_))

# 1. Calculate the Confusion Matrix
cm = confusion_matrix(y_encoded, y_pred_classes)

# 2. Plot the Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)

# 3. Add Labels and Title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of Hand Signs')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# 4. Show the plot
plt.tight_layout() # Adjusts plot to ensure labels fit
plt.show()