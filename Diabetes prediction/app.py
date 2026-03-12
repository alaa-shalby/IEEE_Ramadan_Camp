from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

FINAL_ORDER = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() 
        df_input = pd.DataFrame([data])
        
       
        cat_cols = ['gender', 'smoking_history']
        df_input[cat_cols] = encoder.transform(df_input[cat_cols])
        
        num_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        df_input[num_cols] = scaler.transform(df_input[num_cols])
        
        df_input = df_input[FINAL_ORDER]
        
        prediction = model.predict(df_input)[0]
        
        result = "Positive (Diabetes)" if prediction == 1 else "Negative (No Diabetes)"
        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "result": result
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)