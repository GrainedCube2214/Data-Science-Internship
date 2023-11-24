# stockmarketpred.py
from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__, template_folder='templates')

# Load the trained machine learning model
model = joblib.load('stock_model.pkl')  # Adjust the model file name if needed

def cleanup(df):
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Define the preprocessing function
def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])
    df = cleanup(df)
        
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y', errors='coerce')
    
    #Feature Engineering
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    #df['DayofWeek'] = df['Date'].dt.dayofweek
    df['DayofYear'] = df['Date'].dt.dayofyear
    
    #df['Open-Delta'] = df['Open'] - df['Open'].shift(1)
    """df['Open-Close'] = df['Open'] - df['Close']
    df['High-Low'] = df['High'] - df['Low']
    df['High-Delta'] = df['High'] - df['High'].shift(1)
    df['Low-Delta'] = df['Low'] - df['Low'].shift(1)
    df['Close-Open'] = df['Close'] - df['Open'].shift(1)
    df['Close-Delta'] = df['Close'] - df['Close'].shift(1)"""
    df['Volume-Delta'] = df['Shares Traded'] - df['Shares Traded'].shift(1)
    df['Turnover-Delta'] = df['Turnover (Crores)'] - df['Turnover (Crores)'].shift(1)
    
    #Replace NA values with 0 in the new columns
    df.fillna(0, inplace=True)
    df = cleanup(df)
    
    columns_to_select = ['Turnover-Delta', 'Open', 'DayofYear', 'Day', 'Volume-Delta', 'Shares Traded', 'Month', 'Year', 'Turnover (Crores)']
    df = df[columns_to_select]

    return df

# Define a route for the home page
@app.route('/')
def home():
    return render_template('stockmarketpred.html', prediction=None)  # Pass 'None' initially for prediction

# Define a route for handling the form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = {
            #'Gender': request.form['gender'],
            'Date' : request.form['date'],
            'Open' : request.form['open'],
            'Shares Traded' : request.form['shares_traded'],
            'Turnover (Crores)' : request.form['turnover'],
            # Add other form fields here corresponding to your dataset columns
        }
        print("Got it!")
        # Preprocess the input data
        input_data = preprocess_input(data)
        
        print(data)

        # Make predictions using the loaded model
        prediction = model.predict(input_data)
        print("Ans:",prediction,type(prediction))
        
        #Convert given floating point hours into HH:MM:SS format
        answer = f"High: {prediction[0, 0]:.3f};    Low: {prediction[0, 1]:.3f};    Closing: {prediction[0, 2]:.3f}"

        # Pass the prediction to the HTML template
        return render_template('stockmarketpred.html', prediction=answer)

if __name__ == '__main__':
    app.run(debug=True)
