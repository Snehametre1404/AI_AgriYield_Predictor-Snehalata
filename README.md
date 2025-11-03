# 🌾 AI AgriYield Predictor

**An AI-powered system that predicts crop yield based on soil health, weather, and environmental factors.**  
This project leverages machine learning and data analysis to assist farmers, agronomists, and policymakers in improving agricultural productivity through data-driven insights.

---

## 🚀 Overview

The **AI AgriYield Predictor** analyzes soil, crop, and weather parameters to forecast expected yield.  
It combines Exploratory Data Analysis (EDA), Feature Engineering, and Machine Learning to build an accurate prediction model.  
A Flask-based web interface enables users to upload input data and get instant predictions.

---

## 🧠 Key Features

- 📊 **Data Processing & Cleaning:** Handles missing values, outliers, and normalization of agricultural data.  
- 🌾 **Model Training:** Uses regression and ensemble learning algorithms to predict yield accurately.  
- 🔍 **Feature Importance:** Highlights key soil and climate parameters influencing yield.  
- 🧩 **Visualization Dashboards:** Displays EDA and prediction comparisons (actual vs. predicted).  
- 🌐 **Web App Interface:** Built using Flask for easy interaction and deployment.  
- ☁️ **Deployment Ready:** Includes Procfile and Requirements.txt for seamless cloud deployment (e.g., Render, Heroku).  

---



| Category             | Technologies                     |
| -------------------- | -------------------------------- |
| **Frontend**         | HTML5, CSS3 (Custom + Bootstrap) |
| **Backend**          | Flask (Python)                   |
| **Machine Learning** | Scikit-learn, XGBoost            |
| **Data Handling**    | Pandas, NumPy                    |
| **Visualization**    | Matplotlib, Seaborn              |
| **Version Control**  | Git, GitHub                      |
| **Deployment**       | Render / Heroku (Procfile ready) |


🔬 Model Workflow:

1.Data Collection: Soil, weather, and crop datasets are merged and cleaned.

2.Feature Engineering: Feature scaling and selection using correlation analysis.

3.Model Training: Algorithms like Linear Regression, RandomForest, and XGBoost are trained.

4.Evaluation: Metrics include RMSE, MAE, and R².

5.Prediction: Final model predicts yield for new unseen data.

6.Deployment: Integrated with Flask for real-time prediction via web interface.

💻 Usage Instructions:
   1️⃣ Clone the repository:
   
       1. git clone https://github.com/Snehametre1404/AI_AgriYield_Predictor-Snehalata.git
       
       2. cd AI_AgriYield_Predictor-Snehalata
       
   2️⃣ Install dependencies:
   
        pip install -r app/Requirements.txt
        
   3️⃣ Run the Flask app:
   
       python app/app.py
        
      Then open your browser and navigate to:http://127.0.0.1:5000/

  ## 🚀 Live Deployment

The **AI AgriYield Predictor** web app is live and accessible here:

🔗 **[AI AgriYield Predictor – Live App](https://ai-agriyield-predictor-snehalata.onrender.com)**

This Flask-based web app predicts crop yield using environmental and soil data through a trained machine learning model.



📈 Sample Output:
   1. After uploading soil and weather parameters, the system displays:
   2. Predicted crop yield in tons/hectare

🧾 License:
   This project is licensed under the MIT License.
   See the LICENSE file for details.

👩‍💻 Author:

    - Snehalata 
    
    - MCA (2nd Year) Student
    
    - snehametre333@gmail.com
    
    - GitHub Profile:[Snehametre1404](https://github.com/Snehametre1404)

💡 Future Enhancements

        1.Integration of live weather API data.
        
        2.Adding support for multiple regional crop datasets.
        
        3.Predicting soil fertility index and irrigation advice.
        
        4.Deploying with a React or Streamlit frontend.
 
⭐ Acknowledgments

Special thanks to:

    Open-source libraries: Scikit-learn, Pandas, Flask
    
    Agricultural open datasets used for model training
    
    Mentors and contributors who guided this project   


    


