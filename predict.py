import pandas as pd
import joblib

def predict_price(input_dict):
    model = joblib.load('linear_model.pkl')
    
    df = pd.DataFrame([input_dict])

    for col in ['location_Montego Bay', 'location_Negril', 'location_Ocho Rios', 'location_Portland']:
        if col not in df.columns:
            df[col] = 0

    prediction = model.predict(df)[0]
    return round(prediction, 2)