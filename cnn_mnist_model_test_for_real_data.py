import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
# import tensorflow as tf
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K
from myutils import pre_process_image
from myutils import show_prediction
# K.set_image_dim_ordering('th')

with open('cnn_mnist_model_for_mnist_digit.json','r') as json_file:
	loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights('cnn_mnist_model_for_mnist_digit.h5')
# model = tf.keras.models.load_model("cnn_mnist_model_for_mnist_digit.h5")
print("Loaded saved model")

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

img_gray = pre_process_image(filename="codei.JPG")

img = np.array(img_gray)
img = img.reshape(1, 1, 28, 28).astype('float32')
img = img/255

predicted = model.predict_classes(img)
print(predicted)
show_prediction("codei.JPG",predicted[0])