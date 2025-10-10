import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("crop_yield_dataset.csv")

# 1. Convert Date column to datetime and extract features
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df.drop(columns=['Date'], inplace=True)

# 2. Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)  # Numerical missing values
df.fillna('Unknown', inplace=True)                  # Categorical missing values

# 3. Encode categorical columns
le_crop = LabelEncoder()
le_soil = LabelEncoder()
df['Crop_Type'] = le_crop.fit_transform(df['Crop_Type'])
df['Soil_Type'] = le_soil.fit_transform(df['Soil_Type'])

# 4. Feature Engineering
df['NPK_Ratio'] = df['N'] / (df['P'] + df['K'])  # Nutrient balance
df['Fertility_Index'] = df['N'] + df['P'] + df['K'] + df['Soil_Quality']  # Combined fertility
# Season feature based on Month
def get_season(month):
    if month in [1,2,3,10,11,12]:
        return 1  # Rabi
    elif month in [4,5]:
        return 2  # Summer
    else:
        return 3  # Kharif
df['Season'] = df['Month'].apply(get_season)

# 5. Define features (X) and target (y)
X = df.drop(columns=['Crop_Yield'])
y = df['Crop_Yield']

# 6. Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 8. Save preprocessed data
df.to_csv("crop_yield_preprocessed.csv", index=False)
print("✅ Preprocessed file saved as 'crop_yield_preprocessed.csv'")

# 9. Optional: Save scaled features and target separately for ML use
import numpy as np
np.save("X_scaled.npy", X_scaled)
np.save("y.npy", y.to_numpy())
print("📂 Scaled features and target saved as 'X_scaled.npy' and 'y.npy'")

# 10. Quick check
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
