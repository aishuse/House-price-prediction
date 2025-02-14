import streamlit as st
import numpy as np
import pickle
import json
from PIL import Image

with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

locations = data_columns[3:]

def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(data_columns))
    x[0], x[1], x[2] = sqft, bath, bhk

    if location in locations:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    return round(model.predict([x])[0], 2)

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏡 Bangalore House Price Prediction")

image = Image.open("house.jpg")
st.image(image, use_container_width=True)

location = st.selectbox("Select Location", locations)
sqft = st.number_input("Enter Total Square Feet", min_value=500, max_value=10000, value=1000)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
bhk = st.number_input("Number of BHK", min_value=1, max_value=10, value=3)

if st.button("Predict Price"):
    price = predict_price(location, sqft, bath, bhk)
    st.success(f"💰 Estimated Price: ₹ {price} Lakhs")

st.markdown("<h3 style='text-align: right; color: blue;'>Made with ❤️ by Aiswarya</h3>", unsafe_allow_html=True)
