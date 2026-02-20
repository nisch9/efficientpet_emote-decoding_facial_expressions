import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers

st.cache(allow_output_mutation=True)
CLASS_NAMES = ['Angry', 'Normal', 'Sad', 'Happy']

def load_model():
  img_size = (224, 224)
  channels = 3
  img_shape = (img_size[0], img_size[1], channels)
  base_model = tf.keras.applications.efficientnet.EfficientNetB5(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')
  base_model.trainable = False

  model = Sequential([
      base_model,
      BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
      Dense(256, activation='relu'),
      Dense(128, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),
                  bias_regularizer= regularizers.l1(0.006), activation= 'relu'),
      Dropout(rate= 0.45, seed= 123),
      Dense(4, activation= 'softmax')
  ])
  model.load_weights('./Model/EfficientPet.h5')
  return model

model = load_model()


st.write("""# Pet Emotion Detection""")

file = st.file_uploader("Please upload your pet's face", type=["jpg", "png"])

def import_and_predict(image_data, model):
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img_reshape = np.expand_dims(image,0)
        prediction = model.predict(img_reshape)
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[index]
    confidence = np.max(predictions[0])
    st.write(predicted_class)
    st.write(confidence)
    print(
    "This image most likely belongs to {} face with a {:.2f} percent confidence."
    .format(predicted_class, 100 * np.max(confidence))
)