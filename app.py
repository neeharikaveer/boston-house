from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert form inputs to float values
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)[0]
        return render_template('index.html', prediction_text=f'Estimated House Price: ${prediction * 1000:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
 