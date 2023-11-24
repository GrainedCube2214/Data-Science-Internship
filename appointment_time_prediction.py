# appointment_time_prediction.py
from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__, template_folder='templates')

# Load the trained machine learning model
model = joblib.load('regression_model.pkl')  # Adjust the model file name if needed

# Define the preprocessing function
def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])
    df.dropna(inplace=True)
    df['Alcoholism'] = df['Alcoholism'].apply(lambda x: 1 if x.lower()=="yes" or x.lower()=="y"  else 0)
    df['Hipertension'] = df['Hipertension'].apply(lambda x: 1 if x.lower()=="yes" or x.lower()=="y"  else 0)
    df['Diabetes'] = df['Diabetes'].apply(lambda x: 1 if x.lower()=="yes" or x.lower()=="y"  else 0)
    df['Cancelled'] = df['Cancelled'].apply(lambda x: 1 if x.lower()=="yes" or x.lower()=="y"  else 0)
    # Converting to datetime format
    df['Scheduled_Date'] = pd.to_datetime(df['Scheduled_Date'])
    df['Appointment_Date'] = pd.to_datetime(df['Appointment_Date'])

    df['Day_Difference'] = (df['Appointment_Date'] - df['Scheduled_Date']).dt.days
    df['Day_Difference'] = df['Day_Difference'].astype(int)
    
    # Feature extraction
    df['Scheduled_Year'] = df['Scheduled_Date'].dt.year
    df['Scheduled_Month'] = df['Scheduled_Date'].dt.month
    df['Scheduled_Day'] = df['Scheduled_Date'].dt.day
    df.drop(['Scheduled_Date'], axis=1, inplace=True)
    
    df['Appointment_Month'] = df['Appointment_Date'].dt.month
    df['Appointment_Day'] = df['Appointment_Date'].dt.day
    df.drop(['Appointment_Date'], axis=1, inplace=True)
       
    #Convert all int64 features to float64
    for col in df.columns[df.dtypes == np.int64]:
        df[col] = df[col].astype(np.float64)
    
    # Drop duplicates and remaining null values
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True) 
    
    #Customize column order
    df = df[['Cancelled', 'Appointment_Month', 'Scheduled_Month', 'Day_Difference', 
             'Diabetes', 'Scheduled_Day', 'Hipertension', 'Alcoholism', 'Scheduled_Year', 
             'Appointment_Day', 'Age', 'Appointment_Hour', 'Rate_Of_Cancellation']]
    
    print(df)
    
    return df

# Define a route for the home page
@app.route('/')
def home():
    return render_template('appointment_time_prediction.html', prediction=None)  # Pass 'None' initially for prediction

# Define a route for handling the form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = {
            #'Gender': request.form['gender'],
            'Scheduled_Date': request.form['scheduled_date'],
            'Appointment_Date': request.form['appointment_date'],
            'Appointment_Hour': request.form['appointment_hour'],
            'Age': request.form['age'],
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
        
        #Convert given floating point hours into HH:MM:SS format
        prediction_hours = int(prediction)
        prediction_minutes = int((prediction - prediction_hours) * 60)
        prediction_seconds = int(((prediction - prediction_hours) * 60 - prediction_minutes) * 60)
        prediction_time = f"{prediction_hours:02d}:{prediction_minutes:02d}:{prediction_seconds:02d}"

        # Pass the prediction to the HTML template
        return render_template('appointment_time_prediction.html', prediction=prediction_time)

if __name__ == '__main__':
    app.run(debug=True)
