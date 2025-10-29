import pandas as pd

# ✅ Load dataset
df = pd.read_csv("crop_yield_preprocessed.csv")

# ✅ Keep only selected important columns
selected_columns = [
    'Temperature',
    'Humidity',
    'Soil_Quality',
    'Crop_Type',
    'NPK_Ratio',
    'Fertility_Index',
    'Crop_Yield'
]

# ✅ Filter dataset
df_selected = df[selected_columns]

# ✅ Save filtered dataset
df_selected.to_csv("crop_yield_predicted.csv", index=False)

print("✅ Filtered dataset saved as 'crop_yield_predicted.csv'")
print(df_selected.head())
