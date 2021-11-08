import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import keras
from PIL import Image
import numpy as np

st.title("Image Classification with Inception V3")
st.header("Dog Breed Classification")
st.text("Upload a dog image to obtain its breed")


my_content = open("dogs_name.txt", "r")
dog_names = my_content.read()
dogs_list = dog_names.split('\n')
my_content.close()

def image_classifier(im, weights_file):
  model = keras.models.load_model(weights_file)
  img = img_to_array(im)
  img = Image.resize(img,(224,224))
  img = img.reshape(1,224,224,3)
  predictions = model.predict(img)
  predictions = tf.nn.softmax(predictions)
  predictions = np.argmax(predictions)
  return dogs_list[predictions]

uploaded_file = st.file_uploader("Choose a Dog Image ...", type="jpg")

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption='Uploaded image.', use_column_width=True)
  st.write("")
  st.write("Classifying...")
  label = image_classifier(image, 'my_model.h5')
  st.write(label)