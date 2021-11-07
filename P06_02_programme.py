import gradio as gr
import numpy as np
import tensorflow as tf
import cv2


model = tf.saved_model.load('my_model.h5')
def image_classifier(im):
  im = img_to_array(im)
  im = cv2.resize(im,(224,224))
  im = im.reshape(1,224,224,3)
  predictions = model.predict(img)
  score = tf.nn.softmax(predictions[0])
  return {labels[i]: float(prediction[i]) for i in range(1000)}

iface = gr.Interface(
    image_classifier, 
    gr.inputs.Image(shape=(224, 224)), 
    gr.outputs.Label(score),
    capture_session=True,
    interpretation="default"
    )

if __name__ == "__main__":
    iface.launch()