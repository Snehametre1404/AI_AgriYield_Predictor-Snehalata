from flask import Flask, request, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# -----------------------------
# ✅ Load model from parent 'model' folder
# -----------------------------
# Construct full path (works both locally and on Render)
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "crop_yield_model.pkl")
model_path = os.path.abspath(model_path)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

model = joblib.load(model_path)

# -----------------------------
# ✅ Home route
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# ✅ Predict route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values safely
        Temperature = float(request.form.get("Temperature", 0))
        Humidity = float(request.form.get("Humidity", 0))
        Soil_Quality = float(request.form.get("Soil_Quality", 0))
        NPK_Ratio = float(request.form.get("NPK_Ratio", 0))
        Fertility_Index = float(request.form.get("Fertility_Index", 0))
        Crop_Type = float(request.form.get("Crop_Type", 0))

        # Prepare input data
        input_data = np.array([[Temperature, Humidity, Soil_Quality,
                                NPK_Ratio, Fertility_Index, Crop_Type]])

        # Predict
        prediction = model.predict(input_data)[0]
        prediction = max(0, float(prediction))  # Prevent negative values

        # Display output
        prediction_text = f"🌾 Predicted Crop Yield: {round(prediction, 2)} tons/hectare"
        return render_template("index.html", prediction_text=prediction_text)

    except Exception as e:
        # Handle runtime errors gracefully
        return render_template("index.html", prediction_text=f"⚠️ Error: {str(e)}")

# -----------------------------
# ✅ Run Flask app
# -----------------------------
if __name__ == "__main__":
    # Use environment port for Render, default 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
