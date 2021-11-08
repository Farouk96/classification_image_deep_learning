import gradio as gr
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import keras
import cv2
import numpy as np


my_content = open("dogs_name.txt", "r")
dog_names = my_content.read()
dogs_list = dog_names.split('\n')
my_content.close()
print(dogs_list)

model = keras.models.load_model("my_model.h5")

def image_classifier(im):
  img = img_to_array(im)
  img = cv2.resize(img,(224,224))
  img = img.reshape(1,224,224,3)
  predictions = model.predict(img)
  predictions = tf.nn.softmax(predictions)
  predictions = np.argmax(predictions)
  return dogs_list[predictions]

iface = gr.Interface(
    image_classifier, 
    gr.inputs.Image(shape=(224, 224)), 
    gr.outputs.Label(num_top_classes=3),
    capture_session=True,
    interpretation="default")

if __name__ == "__main__":
    iface.launch(share=True)