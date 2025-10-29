# 📦 Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# ✅ 1. Load Dataset
df = pd.read_csv("planthealth_30000.csv")

# ✅ 2. Drop Unnecessary Columns
if 'Plant_ID' in df.columns:
    df.drop(columns=['Plant_ID'], inplace=True)

# ✅ 3. Handle Timestamp (if present)
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['Year'] = df['Timestamp'].dt.year
    df['Month'] = df['Timestamp'].dt.month
    df['Day'] = df['Timestamp'].dt.day
    df.drop(columns=['Timestamp'], inplace=True)

# ✅ 4. Fill Missing Values
df.fillna(df.mean(numeric_only=True), inplace=True)
df.fillna('Unknown', inplace=True)

# ✅ 5. Encode Categorical Columns
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# ✅ 6. Define Features and Target
target_col = 'Plant_Health_Status'   # Change if your target name is different
X = df.drop(columns=[target_col])
y = df[target_col]

# ✅ 7. Scale Numerical Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ 8. Split into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ 9. Save Preprocessed Data
df.to_csv("plant_health_preprocessed.csv", index=False)
print("✅ Preprocessed file saved as 'plant_health_preprocessed.csv'")

# ✅ 10. Quick Check
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
