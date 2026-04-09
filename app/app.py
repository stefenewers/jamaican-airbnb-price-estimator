import streamlit as st
import pandas as pd
import joblib

model = joblib.load('src/linear_model.pkl')

st.title("🏝️ Jamaican Airbnb Price Estimator")

# Input fields
bedrooms = st.slider("Bedrooms", 1, 5, 2)
bathrooms = st.slider("Bathrooms", 1, 3, 1)
accommodates = st.slider("Accommodates", 1, 8, 2)
has_wifi = st.checkbox("WiFi", True)
has_pool = st.checkbox("Pool", False)
has_ac = st.checkbox("Air Conditioning", True)
has_kitchen = st.checkbox("Kitchen", True)
has_parking = st.checkbox("Parking", True)
location = st.selectbox("Location", ['Kingston', 'Montego Bay', 'Negril', 'Ocho Rios', 'Portland'])

# Format input
input_data = {
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'accommodates': accommodates,
    'has_wifi': int(has_wifi),
    'has_pool': int(has_pool),
    'has_ac': int(has_ac),
    'has_kitchen': int(has_kitchen),
    'has_parking': int(has_parking),
}

# These are the exact dummy columns the model was trained with
location_dummies = ['location_Montego Bay', 'location_Negril', 'location_Ocho Rios', 'location_Portland']

# Create a dictionary with all locations set to 0
location_data = {col: 0 for col in location_dummies}

# If the selected location is one of the dummies, set it to 1
selected_dummy = f'location_{location}'
if selected_dummy in location_data:
    location_data[selected_dummy] = 1
# If location is Kingston (which was dropped during training), do nothing

# Merge the dictionaries
input_data.update(location_data)

# Predict
if st.button("Predict Price"):
    prediction = model.predict(pd.DataFrame([input_data]))[0]
    st.success(f"Estimated Price: **${prediction:.2f} USD/night**")