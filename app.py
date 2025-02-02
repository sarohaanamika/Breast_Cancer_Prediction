from flask import Flask, render_template, request, flash
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
model_path = 'ensemble_model.pkl'
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
        # Check if the loaded model has a 'predict' method
        if not hasattr(model, 'predict'):
            raise TypeError("The loaded object is not a valid model.")
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None  # Handle this gracefully if needed

# Initialize the scaler
scaler = MinMaxScaler()

# Mock fit the scaler
scaler.fit(np.zeros((1, 100)))  # Replace with actual data for real scaling

app = Flask(__name__)
app.secret_key = 'secret_key'  # For flash messages

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None  # Initialize result variable to be passed to template
    if request.method == 'POST':
        try:
            # Extract and process input values in the new format
            input_features = [
                'texture_mean', 'perimeter_mean', 'smoothness_mean', 'concavity_mean',
                'concave_points_mean', 'texture_worst', 'perimeter_worst',
                'smoothness_worst', 'concavity_worst', 'concave_points_worst'
            ]
            
            # Get values from the form and convert them to float
            inputs = np.array([[float(request.form[feature]) for feature in input_features]])

            # Scale the input values
            inputs_scaled = scaler.transform(inputs)

            # Check if the model was loaded successfully before prediction
            prediction = model.predict(inputs_scaled)
            result = 'Malignant' if prediction[0] == 1 else 'Benign'

        except ValueError:
            flash('Please enter valid numerical values for all features.')

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
