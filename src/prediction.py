from tensorflow import keras
import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model_path = '../model/wide_resnet_cifar100.keras'
saved_model = keras.models.load_model(model_path)

IMG_SIZE = (32,32)
N_CLASSES = 100

def preprocess_img(img):
    img = image.smart_resize(img, IMG_SIZE)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')/255.0
    
    return img

def make_prediction(img):
    img_preproced = preprocess_img(img)
    prediction = saved_model.predict(img_preproced)
    predicted_class_labels = np.argmax(prediction, axis=1)
    
    return predicted_class_labels