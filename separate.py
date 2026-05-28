import pandas as pd

# Load merged dataset
df = pd.read_csv("C:/Users/hiine/Desktop/FYP/FYP_dataset/hand_landmarks_all.csv")

# Separate features and labels
X = df.drop("label", axis=1)   # All landmark columns
y = df["label"]                # Gesture labels

# Verify shapes
print("Feature shape:", X.shape)
print("Label shape:", y.shape)

# Verify number of classes
print("Number of gesture classes:", y.nunique())
print("Gesture labels:", y.unique())

# Check for missing values
print("Missing values in X:", X.isnull().sum().sum())
print("Missing values in y:", y.isnull().sum())

# Check number of unique classes
print("Number of gesture classes:", y.nunique())
print("Gesture classes:", y.unique())
