from flask import Flask, jsonify , render_template ,request
import joblib
import numpy as np
app = Flask(__name__)

model=joblib.load("Model.pkl")
scaler=joblib.load("Scaler.pkl")
encoder=joblib.load("Encoder.pkl")

full_data=['age', 'sex', 'bmi', 'children', 'smoker', 'region']
cat_data=['sex','smoker','region']
num_data=[col for col in full_data if col not in cat_data]


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
     data=request.get_json()
     num_col=[data[col] for col in num_data]
     cat_cols=[data[col] for col in cat_data]

     scaled_data=scaler.transform([num_col])

     encoded_data=encoder.transform([cat_cols])

     final_features = np.concatenate([scaled_data, encoded_data], axis=1)

     prediction=model.predict(final_features)
     return jsonify({"Pridiction is":round(float(prediction[0]), 2)})
 
if __name__ == '__main__':
    app.run(debug=True)