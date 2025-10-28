import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

IMG_SIZE = (32,32)
N_CLASSES = 100

def preprocess_img(img):
    img = image.smart_resize(img, IMG_SIZE)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')/255.0
    
    return img