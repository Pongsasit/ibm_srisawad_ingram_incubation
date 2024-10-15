import streamlit as st
import requests
import json

with open('data_use.json', 'r') as file:
    data = json.load(file)

st.title("Price Prediction")
# Input fields for vehicle information

make_options = data['Make']
make = st.selectbox('Choose a brand option:', make_options)

model_options = data['Model']
model = st.selectbox('Choose a make option:', model_options)

year_options = data['Year']
year =  st.selectbox('Choose a year option:', year_options)

engine_fuel_type_options = data['Engine Fuel Type']
engine_fuel_type = st.selectbox('Choose an engine fuel type option:', engine_fuel_type_options)

engine_hp_options = data['Engine HP']
engine_hp = st.selectbox('Choose an engine horse power:', engine_hp_options)

engine_cylinders_options = data['Engine Cylinders']
engine_cylinders = st.selectbox('Choose an engine cylinder option:', engine_cylinders_options)

transmission_type_options = data['Transmission Type']
transmission_type = st.selectbox('Choose an transmission type option:', transmission_type_options)

driven_wheels_options = data['Transmission Type']
driven_wheels = st.selectbox('Choose an transmission type option:', driven_wheels_options)

number_of_doors_options =data['Number of Doors']
number_of_doors = st.selectbox('Choose an transmission type option:', transmission_type_options)

vehicle_size_options =data['Vehicle Size']
vehicle_size = st.selectbox('Choose an vehicle size type option:', vehicle_size_options)


vehicle_style_options =data['Vehicle Style']
vehicle_style = st.selectbox('Choose an vehicle style type option:', vehicle_style_options)


highway_mpg_option = data['highway MPG']
highway_mpg = st.selectbox('Choose an highway mpg option:', highway_mpg_option)

city_mpg_option = data['city mpg']
city_mpg = st.selectbox('Choose an city mpg option:', city_mpg_option)


popularity_option = data['Popularity']
popularity = st.selectbox('Choose an popularity option:', popularity_option)

age_option = data['Years Of Manufacture']
age = st.selectbox('Choose an age option:', age_option)

