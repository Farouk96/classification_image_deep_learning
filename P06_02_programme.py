import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import keras
from PIL import Image
import cv2
import numpy as np

st.title("Image Classification with Inception V3")
st.header("Dog Breed Classification")
st.text("Upload a dog image to obtain its breed")


my_content = open("dogs_name.txt", "r")
dog_names = my_content.read()
dogs_list = dog_names.split('\n')
my_content.close()

def image_classifier(img, weights_file):
  model = keras.models.load_model(weights_file)
  image = cv2.resize(img,(224,224))
  image = image.reshape(1,224,224,3)
  predictions = model.predict(image)
  predictions = tf.nn.softmax(predictions)
  predictions = np.argmax(predictions)
  return dogs_list[predictions]

uploaded_file = st.file_uploader("Choose a Dog Image ...", type="jpg")

if uploaded_file is not None:
  img = Image.open(uploaded_file)
  st.image(img, caption='Uploaded image.', use_column_width=True)
  st.write("")
  st.write("Classifying...")
  label = image_classifier(img, 'my_model.h5')
  st.write(label)