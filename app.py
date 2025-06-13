from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# List of feature column names (order should match model training)
feature_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
                   'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect the form data and keep it in a dictionary to preserve input
        input_data = {col: request.form.get(col) for col in feature_columns}
        features = [float(input_data[col]) for col in feature_columns]
        final_features = [np.array(features)]

        # Predict house price
        prediction = model.predict(final_features)[0]
        predicted_price = f'Estimated House Price: ${prediction * 1000:.2f}' 

        return render_template('index.html',
                               prediction_text=predicted_price,
                               input_data=input_data)
    except Exception as e:
        return render_template('index.html',
                               prediction_text=f'Error: {e}',
                               input_data=request.form)

if __name__ == '__main__':
    app.run(debug=True)
