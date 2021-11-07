import gradio as gr
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, get_file
import keras
import cv2
import numpy as np
from os.path import dirname, realpath, join

# Load human-readable labels for ImageNet.
current_dir = dirname(realpath(__file__))
with open(join(current_dir, "dog_names.json")) as labels_file:
    labels = json.load(labels_file)

model = keras.models.load_model("my_model.h5")

def image_classifier(im):
  img = img_to_array(im)
  img = cv2.resize(img,(224,224))
  img = img.reshape(1,224,224,3)
  predictions = model.predict(img)
  predictions = tf.nn.softmax(predictions[0])
  predictions = np.argmax(predictions)
  return {labels[i]: float(predictions[i]) for i in range(120)}

iface = gr.Interface(
    image_classifier, 
    gr.inputs.Image(shape=(224, 224)), 
    gr.outputs.Label(num_top_classes=3),
    capture_session=True,
    interpretation="default")

if __name__ == "__main__":
    iface.launch()