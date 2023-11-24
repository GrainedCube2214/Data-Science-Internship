# noshow_classification_flask.py
from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__, template_folder='templates')

# Load the trained machine learning model
model = joblib.load('noshow_classification_model.pkl')  # Adjust the model file name if needed

# Define the preprocessing function
def preprocess_input(data):
    # Assume 'data' is a dictionary containing user input
    input_df = pd.DataFrame([data])
    
    input_df['Gender'] = input_df['Gender'].apply(lambda x: 1 if x.upper()=="M" or x.upper()=="MALE" else 0)

    # Convert Scheduled_Date and Appointment_Date to datetime format
    input_df['Scheduled_Date'] = pd.to_datetime(input_df['Scheduled_Date'])
    input_df['Appointment_Date'] = pd.to_datetime(input_df['Appointment_Date'])
    input_df['Alcoholism'] = input_df['Alcoholism'].apply(lambda x: 1 if x.lower()=="yes" or x.lower()=="y"  else 0)
    input_df['Hipertension'] = input_df['Hipertension'].apply(lambda x: 1 if x.lower()=="yes" or x.lower()=="y"  else 0)
    input_df['Diabetes'] = input_df['Diabetes'].apply(lambda x: 1 if x.lower()=="yes" or x.lower()=="y"  else 0)
    input_df['Cancelled'] = input_df['Cancelled'].apply(lambda x: 1 if x.lower()=="yes" or x.lower()=="y"  else 0)
    # Feature extraction
    input_df['Day_Difference'] = (input_df['Appointment_Date'] - input_df['Scheduled_Date']).dt.days
    input_df['Scheduled_Year'] = input_df['Scheduled_Date'].dt.year
    input_df['Scheduled_Month'] = input_df['Scheduled_Date'].dt.month
    input_df['Scheduled_Day'] = input_df['Scheduled_Date'].dt.day
    input_df['Appointment_Month'] = input_df['Appointment_Date'].dt.month
    input_df['Appointment_Day'] = input_df['Appointment_Date'].dt.day

    # Drop datetime columns
    input_df = input_df.drop(['Scheduled_Date', 'Appointment_Date'], axis=1)

    # Convert 'Rate_Of_Cancellation' to float
    input_df['Rate_Of_Cancellation'] = input_df['Rate_Of_Cancellation'].astype(float)

    # Reorder columns to match the order during training
    column_order = ['Cancelled', 'Scheduled_Month', 'Scheduled_Year', 'Diabetes', 'Appointment_Day', 'Day_Difference', 'Alcoholism', 'Hipertension', 'Scheduled_Day', 'Appointment_Month', 
                    'Rate_Of_Cancellation', 'Gender']
    
    input_df = input_df[column_order]

    return input_df

# Define a route for the home page
@app.route('/')
def home():
    return render_template('noshow_classification.html', prediction=None)  # Pass 'None' initially for prediction

# Define a route for handling the form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = {
            'Gender': request.form['gender'],
            'Scheduled_Date': request.form['scheduled_date'],
            'Appointment_Date': request.form['appointment_date'],
            'Alcoholism': request.form['alcoholism'],
            'Hipertension': request.form['hipertension'],
            'Diabetes': request.form['diabetes'],
            'Cancelled': request.form['cancelled'],
            'Rate_Of_Cancellation': request.form['rate_of_cancellation']
            # Add other form fields here corresponding to your dataset columns
        }

        # Preprocess the input data
        input_data = preprocess_input(data)

        # Make predictions using the loaded model
        prediction = model.predict(input_data)

        # Convert the prediction to 'Yes' or 'No'
        prediction_text = 'Yes' if prediction[0] == 1 else 'No'

        # Pass the prediction to the HTML template
        return render_template('noshow_classification.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
