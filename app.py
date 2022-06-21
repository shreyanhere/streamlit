import cv2
import numpy as np
import base64
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19,preprocess_input

# set page layout
st.set_page_config(

    page_title="Xray Classification Web-App",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded",
)




st.markdown("<h1 style='text-align: center; color:  LightGray ;'> 💉 Xray Classification </h1>", unsafe_allow_html=True)
st.subheader("Upload an image and check for Pneumonia")
st.sidebar.subheader("Model Name")
models_list = ["VGG19"]
network = st.sidebar.selectbox("Selected  Model", models_list)
model = tf.keras.models.load_model("saved_model/vgg_50.h5")

# component to upload images
uploaded_file = st.sidebar.file_uploader(
    "Choose an image  file ", type=["jpg", "jpeg", "png"]
)


map_dict = {0: 'Bacterial',
            1: 'Normal',
            2: 'Viral'}


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        p = model.predict(img_reshape)
        pred = model.predict(img_reshape).argmax()
        st.subheader("Predicted Label for the Image is {}".format(map_dict [pred]))
        # plot the prediction probability for each category
        fig=plt.figure(figsize = [10,5])   # [width, height]
        plt.style.use('seaborn-darkgrid')
        x = ["Bacterial","Normal","Viral"]
        y = [ p[0][0], p[0][1], p[0][2] ]
        plt.barh(x, y, color='teal')

        ticks_x = np.linspace(0, 1,4)   # (start, end, number of ticks)
        plt.xticks(ticks_x, fontsize=10, family='fantasy', color='black')
        plt.yticks( size=15, color='navy' )
        for i, v in enumerate(y):
           plt.text(v, i, "  "+str((v*100).round(1))+"%", color='blue', va='center', fontweight='bold')

        plt.title('Prediction Probability', family='serif', fontsize=15, style='italic', weight='bold', color='black', loc='center', rotation=0)
        plt.xlabel('Probability', fontsize=12, weight='bold', color='black')
        plt.ylabel('Category', fontsize=12, weight='bold', color='black')
        st.pyplot(fig)
        
