import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("hand_landmarks_all.csv")

# Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder for later use
joblib.dump(label_encoder, "label_encoder.pkl")

# Verify encoding
print("Original labels:", y.unique())
print("Encoded labels:", set(y_encoded))
print("Total classes:", len(label_encoder.classes_))
