# 🌾 Create and Save Scaler for Crop Yield Prediction

import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

# 1️⃣ Set path to your dataset
dataset_path = os.path.join("data", "crop_yield_predicted.csv")

# 2️⃣ Load dataset
try:
    df = pd.read_csv(dataset_path)
    print(f"✅ Dataset loaded successfully from {dataset_path}")
except FileNotFoundError:
    print(f"❌ ERROR: Dataset not found at {dataset_path}")
    exit()

# 3️⃣ Separate features (X) from target (y)
# Replace 'Crop_Yield' with your actual target column name if different
if 'Crop_Yield' not in df.columns:
    print("❌ ERROR: 'Crop_Yield' column not found in dataset")
    exit()

X = df.drop("Crop_Yield", axis=1)

# 4️⃣ Create and fit the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5️⃣ Save the trained scaler to a file
scaler_filename = "scaler.pkl"
pickle.dump(scaler, open(scaler_filename, "wb"))

print(f"✅ Scaler created and saved successfully as {scaler_filename}")

# 6️⃣ Optional: Show first 5 rows of scaled data
print("\n📊 Scaled data sample (first 5 rows):")
print(X_scaled[:5])
