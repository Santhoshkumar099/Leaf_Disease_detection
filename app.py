import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import pickle
import tensorflow as tf
import os
import streamlit as st

st.write("Current working directory:", os.getcwd())
st.write("Files in directory:", os.listdir())
import pickle
import streamlit as st

try:
    with open('leaf_disease_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Model Prediction
def model_prediction(test_image):
    image = Image.open(test_image).resize((128, 128))
    input_arr = np.array(image)[np.newaxis, ...]
    input_arr = input_arr / 255.0  # Normalize the image
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


# Set up home page and option menu
selected = option_menu("Control Panel",
                        options=["Disease Prediction", "ABOUT"],
                        icons=["lightbulb", "info-circle"],
                        default_index=1,
                        orientation="horizontal")

# Set up home page and option menu
selected = option_menu("Control Panel",
                        options=["Disease Prediction", "ABOUT"],
                        icons=["lightbulb", "info-circle"],
                        default_index=1,
                        orientation="horizontal")




#setup the detail for the option 'ABOUT'
if selected == "ABOUT":
        st.header('''Leaf Disease Detection''')
        st.markdown(''' Leaf disease detection involves identifying and diagnosing plant leaf diseases using various methods such as visual inspection, image analysis, and machine learning. High-resolution images of leaves are processed and analyzed with techniques like Convolutional Neural Networks (CNNs) to classify and detect diseases. This helps in timely intervention, optimizing pesticide use, and improving crop yields''',unsafe_allow_html=True)
        st.header('Personal Information')
        Name = (f'{"Name :"}  {"Santhosh Kumar M"}')
        mail = (f'{"Mail :"}  {"sksanthoshhkumar99@gmail.com"}')
        st.markdown(Name)
        st.markdown(mail)
        c1,c2=st.columns(2)
        with c1:
            if st.button('Show Github Profile'):
                st.markdown('[Click here to visit github](https://github.com/Santhoshkumar099)')

        with c2:
            if st.button('Show Linkedin Profile'):
                st.markdown('[Click here to visit linkedin](https://www.linkedin.com/in/santhosh-kumar-2040ab188/)')

        github = (f'{"Github :"}  {"https://github.com/Santhoshkumar099"}')
        linkedin = (f'{"LinkedIn :"}  {"https://www.linkedin.com/in/santhosh-kumar-2040ab188/"}')
        description = "An Aspiring DATA-SCIENTIST..!"

        st.markdown("This project is done by Santhosh Kumar M")


#Prediction Page
if selected == "Disease Prediction":
    # Created form to get the user input 
    with st.form('prediction'):
        st.header("Disease Dectection")
        test_image = st.file_uploader("Choose an Image:")
        button=st.form_submit_button('PREDICT',use_container_width=True)
    if button:
        with st.spinner("Predicting..."):

            st.write("The Disease of the Leaf is")
            result_index = model_prediction(test_image)
            #Reading Labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy']
            st.success(f"Predicted : :green[{class_name[result_index]}]")
