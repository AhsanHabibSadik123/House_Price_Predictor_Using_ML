from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder='../Frontend')

model = joblib.load(os.path.join(os.path.dirname(__file__), 'bangalore_home_price_model.pkl'))
columns = joblib.load(os.path.join(os.path.dirname(__file__), 'model_columns.pkl'))

@app.route('/')
def home():
    # get only location columns (skip sqft, bath, bhk which are the first 3)
    locations = columns[3:]
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    sqft = float(request.form['sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if location in columns:
        loc_index = columns.index(location)
        x[loc_index] = 1
    predicted_price = model.predict([x])[0]

    locations = columns[3:]  # location list for dropdown

    # Pass inputs back to template too
    return render_template('index.html',
                           prediction_text=f'Estimated Price: ₹ {predicted_price:.2f} Lakhs',
                           locations=locations,
                           input_location=location,
                           input_sqft=sqft,
                           input_bath=bath,
                           input_bhk=bhk)


if __name__ == '__main__':
    app.run(debug=True)
