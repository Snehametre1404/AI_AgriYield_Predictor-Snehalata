import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Set working directory
os.chdir(r"D:\Cropyieldintership")
print("Current working directory:", os.getcwd())

# Load dataset
df = pd.read_csv("data/crop_yield_predicted.csv")
print("Dataset loaded:", df.shape)

# Encode Crop_Type if exists
if 'Crop_Type' in df.columns:
    le = LabelEncoder()
    df['Crop_Type_encoded'] = le.fit_transform(df['Crop_Type'])

# Define features and target
X_cols = ['Temperature', 'Humidity', 'Soil_Quality', 'NPK_Ratio', 'Fertility_Index']
if 'Crop_Type_encoded' in df.columns:
    X_cols.append('Crop_Type_encoded')

X = df[X_cols]
y = df['Crop_Yield']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the saved model
model_path = "model/crop_yield_model.pkl"
best_model = joblib.load(model_path)
print("✅ Model loaded successfully")

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate model metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# Save predictions to CSV
if not os.path.exists("model"):
    os.makedirs("model")

results = X_test.copy()
results['Actual_Yield'] = y_test
results['Predicted_Yield'] = y_pred
results.to_csv("model/predictions.csv", index=False)
print("✅ Predictions saved to model/predictions.csv")

# Plot prediction errors
errors = y_test - y_pred
plt.figure(figsize=(8,5))
plt.hist(errors, bins=30, color='orange', alpha=0.7)
plt.title("Prediction Errors Distribution")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.show()

# Actual vs Predicted Scatter Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual Crop Yield")
plt.ylabel("Predicted Crop Yield")
plt.title("Actual vs Predicted Crop Yield")
plt.show()

# Feature Importance for tree-based models
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    print("\nFeature Importance:")
    print(feat_imp)
    
    plt.figure(figsize=(10,6))
    plt.barh(feat_imp["Feature"], feat_imp["Importance"], color="skyblue")
    plt.gca().invert_yaxis()
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.show()
