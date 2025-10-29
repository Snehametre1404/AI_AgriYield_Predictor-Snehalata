from flask import Flask, request, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# -----------------------------
# Load model from parent 'model' folder
# -----------------------------
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "crop_yield_model.pkl")
model_path = os.path.abspath(model_path)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")
model = joblib.load(model_path)

# -----------------------------
# Home route
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Predict route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get numeric inputs
        Temperature = float(request.form["Temperature"])
        Humidity = float(request.form["Humidity"])
        Soil_Quality = float(request.form["Soil_Quality"])
        NPK_Ratio = float(request.form["NPK_Ratio"])
        Fertility_Index = float(request.form["Fertility_Index"])
        Crop_Type = int(request.form["Crop_Type"])

        # Prepare input array
        input_data = np.array([[Temperature, Humidity, Soil_Quality,
                                NPK_Ratio, Fertility_Index, Crop_Type]])

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction = max(0, float(prediction))  # Clip negative values
        prediction_text = f"🌾 Predicted Crop Yield: {round(prediction, 2)} tons/hectare"

        return render_template("index.html", prediction_text=prediction_text)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
