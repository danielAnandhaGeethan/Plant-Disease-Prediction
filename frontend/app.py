import streamlit as st
from tensorflow import keras
from PIL import Image
import cv2
import numpy as np

def load_model(model_path):
    model = keras.models.load_model(model_path)

    return model

def proprocess_image(image):
    target_size = (256, 256)
    
    image_array = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    image_array = cv2.resize(image_array, target_size)

    image_array = image_array / 255.0

    if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)

    return image_array

def predict_disease(model, image):
    class_names = ["Healthy", "Powdery", "Rust"]

    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]

    return predicted_class

def main():
    path = '../backend/model.h5'

    st.markdown("<h1 style='font-size: 29px; text-align: center;'>Plant Disease Prediction</h1>", unsafe_allow_html = True)

    uploaded_image = st.file_uploader("Sprout an image : ")

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image = proprocess_image(image)

        with st.columns(27)[4]:
            st.image(image, caption="Uploaded Image", width = 500)

        answer = ""

        with st.columns(3)[1]:
            btn = st.markdown("<div style='text-align: center; width: 100%;'><button style=' border: none; padding: 10px 20px; font-size: 16px; cursor: pointer; text-color: white;'>Diagnose the Plant</button></div>", unsafe_allow_html = True)


            if btn:
                model = load_model(path)

                answer = predict_disease(model, image)

        st.markdown(f"<h1 style='font-size: 42px; text-align: center'>{answer.capitalize()}</h1>", unsafe_allow_html = True)

if __name__ == "__main__":
    main()