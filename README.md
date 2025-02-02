# Breast_Cancer_Prediction
# Breast Cancer Prediction Web Application
This web application provides a machine learning-based tool for breast cancer prediction using tissue sample features. The application uses a trained ensemble model to classify breast masses as either benign or malignant based on various tissue characteristics.
Features

Web interface for inputting tissue sample measurements
Real-time prediction using a pre-trained machine learning model
Input validation and error handling
Min-Max scaling of input features for consistent predictions

## Project Structure
project/
│
├── app.py                 # Flask web application
├── data_processing.py     # Data preprocessing script
├── ensemble_model.pkl     # Trained machine learning model
├── templates/            
│   └── index.html        # HTML template for web interface
└── README.md             # This file

# Prerequisites

Python 3.7+
Flask
scikit-learn
numpy
pandas
pickle

# Installation

1. Clone the repository:
git clone <repository-url>
cd <project-directory>

2. Install required packages:
pip install flask scikit-learn numpy pandas

3. Ensure the model file ensemble_model.pkl is present in the root directory.

# Usage

1. Start the Flask application:
   python app.py
2. Open a web browser and navigate to http://localhost:5000
3. Enter the following tissue sample measurements:

Texture (mean and worst)
Perimeter (mean and worst)
Smoothness (mean and worst)
Concavity (mean and worst)
Concave points (mean and worst)


4. Click submit to receive the prediction (Benign or Malignant)

# Data Processing
The data_processing.py script handles:

Loading the raw breast cancer dataset
Applying Min-Max scaling to normalize features
Saving the processed data to a new CSV file

# Security Notes

The application includes basic error handling and input validation
The secret key in app.py should be changed before deployment
Debug mode should be disabled in production

# Contributing
Please feel free to submit issues and pull requests for any improvements.   
