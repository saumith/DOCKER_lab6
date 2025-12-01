# ============================================================================
# FILE: src/app.py
# ============================================================================
from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, 
            template_folder=os.path.join(PROJECT_ROOT, 'templates'),
            static_folder=os.path.join(PROJECT_ROOT, 'static'))

# Load the Random Forest model from models directory
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
model = joblib.load(os.path.join(MODEL_DIR, 'diabetes_rf_model.pkl'))
feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get data from form or JSON
            if request.is_json:
                data = request.get_json()
            else:
                data = request.form
            
            # Extract features in the correct order
            pregnancies = float(data['pregnancies'])
            glucose = float(data['glucose'])
            blood_pressure = float(data['blood_pressure'])
            skin_thickness = float(data['skin_thickness'])
            insulin = float(data['insulin'])
            bmi = float(data['bmi'])
            diabetes_pedigree = float(data['diabetes_pedigree'])
            age = float(data['age'])
            
            # Prepare input data (no scaling needed for Random Forest!)
            input_data = np.array([[
                pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, diabetes_pedigree, age
            ]])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            # Get probability of being diabetic
            diabetic_probability = float(probability[1])
            
            # Determine result and risk level
            has_diabetes = prediction == 1
            
            if diabetic_probability < 0.3:
                risk_level = "Low Risk"
            elif diabetic_probability < 0.6:
                risk_level = "Moderate Risk"
            else:
                risk_level = "High Risk"
            
            # Confidence is the maximum probability
            confidence = max(probability[0], probability[1])
            
            return jsonify({
                "prediction": "Diabetic" if has_diabetes else "Non-Diabetic",
                "probability": f"{diabetic_probability * 100:.2f}%",
                "risk_level": risk_level,
                "confidence": f"{confidence * 100:.2f}%",
                "probabilities": {
                    "Non-Diabetic": f"{probability[0] * 100:.2f}%",
                    "Diabetic": f"{probability[1] * 100:.2f}%"
                }
            })
        
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    elif request.method == 'GET':
        return render_template('predict.html')

@app.route('/feature-importance')
def feature_importance():
    """Return feature importance scores"""
    try:
        importance = model.feature_importances_
        features = {
            feature_names[i]: float(importance[i]) 
            for i in range(len(feature_names))
        }
        # Sort by importance
        sorted_features = dict(sorted(features.items(), key=lambda x: x[1], reverse=True))
        return jsonify(sorted_features)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/model-info')
def model_info():
    """Return model information"""
    try:
        info = {
            "model_type": "Random Forest Classifier",
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "n_features": model.n_features_in_,
            "n_classes": int(model.n_classes_),
            "feature_names": feature_names
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model": "Random Forest"}), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)