import numpy as np
import cv2
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K
from myutils import pre_process_image
from myutils import show_multiple_prediction
from myutils import get_subimages

# K.set_image_dim_ordering('th')

img = np.ones((720,1280,3),dtype = np.uint8)*255

drawing = 0

def draw_rectangle_with_drag(event, x, y, flags, param):
      
    global ix, iy, drawing, img
      
    if event == cv2.EVENT_LBUTTONUP:
        drawing = not drawing
        ix = x
        iy = y      
        cv2.circle(img,(ix,iy),3,(0,0,255),-1)      
              
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img,(x,y),6,(0,0,255),-1)  

cv2.namedWindow(winname = "Title of Popup Window")
cv2.setMouseCallback("Title of Popup Window", 
                     draw_rectangle_with_drag)

while True:
    cv2.imshow("Title of Popup Window", img)
      
    if cv2.waitKey(10) == 27:
        break
  
cv2.destroyAllWindows()    

cv2.imwrite('temp.jpg',img)

algo = input('ALPHABET or NUMBER:    ')

if algo == 'a':
    with open('cnn_model_for_real_data.json','r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    model.load_weights("cnn_model_for_real_data.h5")
    print("Loaded saved model")

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    subimages, rects = get_subimages('temp.jpg')

    results = []

    for image in subimages:
        img_gray = pre_process_image(img=image)
        img = np.array(img_gray)
        img = img.reshape(1, 1, 28, 28).astype('float32')
        img = img/255

        predicted = model.predict_classes(img)
        results.append(chr(predicted[0]))
    show_multiple_prediction('temp.jpg',results)
if algo == 'b':
    with open('cnn_mnist_model_for_mnist_digit.json','r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    model.load_weights("cnn_mnist_model_for_mnist_digit.h5")
    print("Loaded saved model")

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    subimages, rects = get_subimages('temp.jpg')

    results = []

    for image in subimages:
        img_gray = pre_process_image(img=image)
        img = np.array(img_gray)
        img = img.reshape(1, 1, 28, 28).astype('float32')
        img = img/255

        predicted = model.predict_classes(img)
        results.append(predicted[0])

    show_multiple_prediction('temp.jpg',results)

