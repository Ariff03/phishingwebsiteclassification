from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from FeatureExtractor import FeatureExtractor

extractor = FeatureExtractor()

# Load trained model
with open('models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('models/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('models/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
with open('models/xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
with open('models/dt_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Ask user to enter the URL
        url = request.form['url']

        # 2. Extract the features from the URL
        features = extractor.extract_features(url)  
        features_array = np.array(list(features.values()), dtype=float).reshape(1, -1)

        X_features = scaler.transform(features_array)
    
        # 3. All five models make their predictions
        predictions = int(xgb_model.predict(X_features)[0])
        # predictions.append(int(rf_model.predict(features_array)[0]))
        # predictions.append(int(svm_model.predict(features_array)[0]))
        # predictions.append(int(knn_model.predict(features_array)[0]))
        # predictions.append(int(xgb_model.predict(features_array)[0]))
        # predictions.append(int(dt_model.predict(features_array)[0]))

        # print("Feature vector:", features)
        # print("Scaled features:", X_features)    
        # print("Prediction:", predictions)

        # 4. Choose the best prediction results
        # final_prediction = max(set(predictions), key=predictions.count)
    
        print(predictions)
        print(features_array)
        
        # 5. Classify whether the URL is phishing or legitimate based on the result
        result = "Phishing" if predictions == 0 else "Legitimate"
    
        # 6. Display the result on the website
        return render_template('index.html', prediction=result)
    
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)