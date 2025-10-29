import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1️⃣ Set working directory
os.chdir(r"D:\Cropyieldintership")
print("Current working directory:", os.getcwd())

# 2️⃣ Load dataset
df = pd.read_csv("data/crop_yield_predicted.csv")
print("Dataset loaded:", df.shape)

# 3️⃣ Encode Crop_Type if it exists
if 'Crop_Type' in df.columns:
    le = LabelEncoder()
    df['Crop_Type_encoded'] = le.fit_transform(df['Crop_Type'])

# 4️⃣ Define features and target
X_cols = ['Temperature', 'Humidity', 'Soil_Quality', 'NPK_Ratio', 'Fertility_Index']
if 'Crop_Type_encoded' in df.columns:
    X_cols.append('Crop_Type_encoded')

X = df[X_cols]
y = df['Crop_Yield']

# 5️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Load model if exists, else raise error
model_path = "model/crop_yield_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please train and save the model first.")
best_model = joblib.load(model_path)
print("Model loaded successfully")



