import streamlit as st
import json
import requests
import base64

with open('data_use.json', 'r') as file:
    data = json.load(file)

# Function to convert uploaded images to base64
def image_to_base64(image):
    if image is not None:
        return base64.b64encode(image.read()).decode('utf-8')
    return None


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

driven_wheels_options = data['Driven_Wheels']
driven_wheels = st.selectbox('Choose an amount of driven wheel option:', driven_wheels_options)

number_of_doors_options =data['Number of Doors']
number_of_doors = st.selectbox('Choose an transmission type option:', number_of_doors_options)

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

# Image upload slots
st.subheader("Upload Vehicle Images")
front_view = st.file_uploader("Upload Front View Image", type=['jpg', 'jpeg', 'png'])
rear_view = st.file_uploader("Upload Rear View Image", type=['jpg', 'jpeg', 'png'])
right_view = st.file_uploader("Upload Right View Image", type=['jpg', 'jpeg', 'png'])
left_view = st.file_uploader("Upload Left View Image", type=['jpg', 'jpeg', 'png'])

# Convert images to base64
front_view_base64 = image_to_base64(front_view)
rear_view_base64 = image_to_base64(rear_view)
right_view_base64 = image_to_base64(right_view)
left_view_base64 = image_to_base64(left_view)

# Button to submit the data for prediction
if st.button('Submit AutoAI Price Prediction'):
    # Create the JSON payload in key-value format
    payload = {
        "Make": make,
        "Model": model,
        "Year": year,
        "Engine Fuel Type": engine_fuel_type,
        "Engine HP": engine_hp,
        "Engine Cylinders": engine_cylinders,
        "Transmission Type": transmission_type,
        "Driven_Wheels": driven_wheels,
        "Number of Doors": number_of_doors,
        "Vehicle Size": vehicle_size,
        "Vehicle Style": vehicle_style,
        "highway MPG": highway_mpg,
        "city mpg": city_mpg,
        "Popularity": popularity,
        "Years Of Manufacture": age,
        "Front View Image": front_view_base64,
        "Rear View Image": rear_view_base64,
        "Right View Image": right_view_base64,
        "Left View Image": left_view_base64
    }

    # Send the JSON payload to the backend API
    try:
        response = requests.post('http://localhost:8080/autoai', json=payload)
        
        if response.status_code == 200:
            st.success('Data sent to backend successfully!')
            response_data = response.json()
            st.write('Backend Response:', response_data)
        else:
            st.error(f'Error: {response.status_code} - {response.text}')
    except Exception as e:
        st.error(f'An error occurred: {e}')