import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# 1️⃣ Load dataset
df = pd.read_csv(r"D:\Cropyieldintership\data\crop_yield_predicted.csv")

# 2️⃣ Load trained model
model = joblib.load("model/crop_yield_model.pkl")

# 3️⃣ Encode Crop_Type if needed
if 'Crop_Type' in df.columns:
    le = LabelEncoder()
    df['Crop_Type_encoded'] = le.fit_transform(df['Crop_Type'])

# 4️⃣ Select features used in training
X_cols = ['Temperature', 'Humidity', 'Soil_Quality', 'NPK_Ratio', 'Fertility_Index']
if 'Crop_Type_encoded' in df.columns:
    X_cols.append('Crop_Type_encoded')

X = df[X_cols]
y_actual = df['Crop_Yield']

# 5️⃣ Make predictions on the dataset
y_pred = model.predict(X)

# 6️⃣ Compare predicted vs actual
comparison = pd.DataFrame({
    'Actual': y_actual,
    'Predicted': y_pred,
    'Difference': y_actual - y_pred
})

print(comparison.head(10))  # Shows first 10 rows