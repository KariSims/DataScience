import streamlit as st
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import numpy as np

def resize_image(image):
    resized_image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
    return resized_image
# Se trouve deja dans la preparation des data dans le model

def predict(image):
    # Loading model
    model = keras.saving.load_model("model_CNN.h5")

    # Resize image into 50x50
    resized_image = resize_image(image)

    # Convert into numpy array
    image_array = np.array(resized_image)

    # Rescale image (0-1)
    rescaled_image = image_array.astype(np.float32) / 255.0

    # Expanding dimensions
    input_image = np.expand_dims(rescaled_image, axis=0)

    # Make prediction
    pred = model.predict(input_image)

    # Get the predict class index
    class_index = np.argmax(pred)

    # Convert the prediction to a label
    if class_index == 0:			#0.5
        label = 'Uninfected Cell'
    else:
        label = 'Parasitized Cell'
    return label


def main():
    # Set background image
    page_bg_img = '''
        <style>
        body {
            background-image: url("https://www.paho.org/sites/default/files/mosquitos_copia.jpg");
            background-size: cover;
            opacity: 0.80;
        }
        </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.title("Application de Detection de la Malaria")
    st.write("Welcome to the Malaria Disease Detection! Please upload an image of a cell, and we will predict if it is parasitized or uninfected. Once the image is uploaded, it will be displayed on the screen. Our model will then make a prediction, and the result will be shown below the image.")

    # File uploader
    uploaded_file = st.file_uploader("Charger votre image avec les formats jpg, jpeg, png...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file :
        # Read image  /is not None
        image = Image.open(uploaded_file)

        # Convert PIL Image to numpy array
        image_array = np.array(image)

        # Display uploaded image
        st.image(image_array, caption='Image chargée')

        # Make prediction
        prediction = predict(image_array)
        prediction = "<h3 style='font-family: Arial;'>Prediction: " + prediction + "</h3>"
        st.write(prediction, unsafe_allow_html=True)

    st.write("Par Prince SIMBA, Fatou KINE et Samsom MUWAWA")

if __name__ == '__main__':
    main()
