from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and artifacts
MODEL_DIR = "models"
DATA_DIR = "data"

model = joblib.load(os.path.join(MODEL_DIR, "house_price_model.pkl"))
feature_names = joblib.load(os.path.join(DATA_DIR, "feature_names.pkl"))
label_encoders = joblib.load(os.path.join(DATA_DIR, "label_encoders.pkl"))

@app.route('/')
def home():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        input_data = []
        for feature in feature_names:
            value = request.form.get(feature)
            
            # Handle categorical features
            if feature in label_encoders:
                value = label_encoders[feature].transform([value])[0]
            else:
                value = float(value)
            
            input_data.append(value)
        
        # Make prediction
        features = np.array(input_data).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        return render_template('result.html', 
                             prediction=f"PKR {prediction:,.2f}")
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        input_data = []
        
        for feature in feature_names:
            value = data.get(feature)
            if value is None:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            
            if feature in label_encoders:
                value = label_encoders[feature].transform([value])[0]
            else:
                value = float(value)
            
            input_data.append(value)
        
        features = np.array(input_data).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        return jsonify({'prediction': float(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)