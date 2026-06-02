from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the pre-trained model pipeline
model = joblib.load('car_price_model.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    car_brand = request.form['car_brand']
    engine_condition = request.form['engine_condition']
    fuel_type = request.form['fuel_type']
    reparation_work = 'Yes' if request.form['reparation_work'] == 'yes' else 'No'
    purchase_value = float(request.form['purchase_value'])
    no_of_buyers = int(request.form['no_of_buyers'])
    kilometer_running = float(request.form['kilometer_running'])
    car_age = int(request.form['car_age'])

    # Create input DataFrame with exact column order as training
    input_data = pd.DataFrame({
        'car_brand': [car_brand],
        'engine_condition': [engine_condition],
        'fuel_type': [fuel_type],
        'reparation_work': [reparation_work],
        'purchase_value': [purchase_value],
        'no_of_buyers': [no_of_buyers],
        'kilometer_running': [kilometer_running],
        'car_age': [car_age]
    })

    # Predict
    prediction = model.predict(input_data)[0]
    predicted_price = round(prediction, 2)

    return render_template('result.html', 
                         car_brand=car_brand,
                         engine_condition=engine_condition,
                         fuel_type=fuel_type,
                         reparation_work=reparation_work,
                         purchase_value=purchase_value,
                         no_of_buyers=no_of_buyers,
                         kilometer_running=kilometer_running,
                         car_age=car_age,
                         predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
