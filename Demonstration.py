# Importing all applications and downloading them using: "pip install ..."

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import tkinter as tk
from tkinter import messagebox
from flask import Flask, request, jsonify

# Loads the dataset
data_cleaned = pd.read_csv('NFLX.csv')

# Makes sure Date, Year and Month is in the correct format.
data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'])
data_cleaned['Year'] = data_cleaned['Date'].dt.year
data_cleaned['Month'] = data_cleaned['Date'].dt.month_name()

# This just converts the catagorys to numeric values using get_dummie
data_encoded = pd.get_dummies(data_cleaned, columns=['Month', 'Year'], drop_first=True)

# Define target variable and selected featuress.
target_variable = 'Close'
selected_features = ['Open', 'High', 'Low', 'Volume'] + [col for col in data_encoded.columns if 'Month_' in col or 'Year_' in col]

# Define target variable and selected featuress.
X = data_encoded[selected_features]
y = data_encoded[target_variable]

# Train the model using LinearRegression algorithm. 
model = LinearRegression()
model.fit(X, y)

# Model is saved to a file
joblib.dump(model, 'linear_regression_model.pkl')

# Prediction Function.
def predict_stock_price(input_data):
    # This loads the Model
    model = joblib.load('linear_regression_model.pkl')

    # Convert input data into format so model can understand it
    input_df = pd.DataFrame([input_data])
    input_df_encoded = pd.get_dummies(input_df, columns=['Month', 'Year'], drop_first=True)

    # Makes usre columns are present
    for col in selected_features:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0
    input_df_encoded = input_df_encoded[selected_features]

    # This makes the prediction
    prediction = model.predict(input_df_encoded)
    return prediction[0]

# Attempt at building GUI with Tkinter (It worked after I fixed it after the presentation, it wasn't working before though). 
def run_tkinter_app():
    def on_predict():
        input_data = {
            'Open': float(open_entry.get()),
            'High': float(high_entry.get()),
            'Low': float(low_entry.get()),
            'Volume': int(volume_entry.get()),
            'Month': month_entry.get(),
            'Year': int(year_entry.get())
        }
        predicted_price = predict_stock_price(input_data)
        messagebox.showinfo("Prediction", f"Predicted Stock Price: {predicted_price}")

    # Setting up the GUI window
    root = tk.Tk()
    root.title("Stock Price Predictor")

    # Input fields
    tk.Label(root, text="Open:").grid(row=0)
    open_entry = tk.Entry(root)
    open_entry.grid(row=0, column=1)

    tk.Label(root, text="High:").grid(row=1)
    high_entry = tk.Entry(root)
    high_entry.grid(row=1, column=1)

    tk.Label(root, text="Low:").grid(row=2)
    low_entry = tk.Entry(root)
    low_entry.grid(row=2, column=1)

    tk.Label(root, text="Volume:").grid(row=3)
    volume_entry = tk.Entry(root)
    volume_entry.grid(row=3, column=1)

    tk.Label(root, text="Month:").grid(row=4)
    month_entry = tk.Entry(root)
    month_entry.grid(row=4, column=1)

    tk.Label(root, text="Year:").grid(row=5)
    year_entry = tk.Entry(root)
    year_entry.grid(row=5, column=1)

    # Prediction button
    predict_button = tk.Button(root, text="Predict", command=on_predict)
    predict_button.grid(row=6, columnspan=2)

    # This Runs the Application
    root.mainloop()

# Flask web application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    prediction = predict_stock_price(input_data)
    return jsonify({"predicted_price": prediction})

if __name__ == '__main__':
    
# Pretty much if you wanna run the tkinter, you can enable the comment so for example instead of #run_tkinter_app(), you just remove the #. Cause last time the tkinter GUI *
# * Wasn't working so I had to use the command line to predict the stock price. But I think it works now. However if you wanna use the old method, you can use this line in *
# * the terminal: Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -Body '{"Open": x, "High": x, "Low": x, "Volume": x, "Month": "x", "Year": x}' -ContentType 'application/json'
# Just make sure to replace x with an interger or the month or year.

    # Run Tkinter GUI
    run_tkinter_app()

    # Run Flask app
    app.run(debug=True)