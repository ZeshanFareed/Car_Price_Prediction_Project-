from django.shortcuts import render
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def user(request):
    return render(request, 'userinput.html')

def viewdata(request):
    # Load the dataset
    df = pd.read_csv("C:/Users/PMLS/Documents/ML/ML Algorithms/car data.csv")
    
    # Encode categorical features
    Car_Name_lb = LabelEncoder()
    df['Car_Name'] = Car_Name_lb.fit_transform(df['Car_Name'])
    
    Fuel_Type_lb = LabelEncoder()
    df['Fuel_Type'] = Fuel_Type_lb.fit_transform(df['Fuel_Type'])
    
    
    Selling_type_lb = LabelEncoder()
    df['Selling_type'] = Selling_type_lb.fit_transform(df['Selling_type'])
    
    Transmission_lb = LabelEncoder()
    df['Transmission'] = Transmission_lb.fit_transform(df['Transmission'])

    # Split the data into input and output
    input_data_x = df.drop(columns=['Selling_Price'])
    output_data_y = df['Selling_Price']
    
    # Scale the features
    ss = StandardScaler()
    input_data_x = pd.DataFrame(ss.fit_transform(input_data_x), columns=input_data_x.columns)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(input_data_x, output_data_y, test_size=0.2, random_state=42)
    
    # Train the model
    RF = RandomForestRegressor()
    RF.fit(X_train, y_train)

    # Get the user input from the GET request
    if request.method == 'GET' and 'Car_Name' in request.GET:
        new_data = {
            'Car_Name': Car_Name_lb.transform([request.GET['Car_Name']])[0],
            'Year': int(request.GET['Year']),
            'Present_Price': float(request.GET['Present_Price']),
            'Driven_kms': int(request.GET['Driven_kms']),
            'Fuel_Type': Fuel_Type_lb.transform([request.GET['Fuel_Type']])[0],
            'Selling_type': Selling_type_lb.transform([request.GET['Selling_type']])[0],
            'Transmission': Transmission_lb.transform([request.GET['Transmission']])[0],
            'Owner': int(request.GET['Owner'])
        }
        new_data_df = pd.DataFrame([new_data])
        new_data_df = ss.transform(new_data_df)
        
        # Make prediction
        y_pred = RF.predict(new_data_df)
        
        data = {
            'message': 'The Predicted Car Price is = ',
            'prediction': y_pred[0]
        }
    else:
        data = {
            'message': '',
            'prediction': None
        }

    return render(request, 'viewdata.html', data)
