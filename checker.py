import pandas as pd

# Load merged dataset
df = pd.read_csv("bim_hand_sign_dataset.csv")

# Check dataset shape
print("Dataset shape:", df.shape)

# Preview first few rows
print(df.head())
