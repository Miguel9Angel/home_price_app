import streamlit as st 
import pandas as pd
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from src.preprocessing import preprocess_image
from src.prediction import make_prediction


st.title('CIFAR-100 Image Classification Model')

cifar100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 
    'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 
    'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

st.markdown("This project addresses the challenge of image classification on the CIFAR-100 dataset (100 distinct object classes), with the primary goal of maximizing model accuracy.")

num_columns = 4
num_classes = len(cifar100_classes)
classes_per_column = num_classes // num_columns

with st.expander("Click to view the complete list of 100 CIFAR-100 Object Classes"):
    st.header("🖼️ 100 Object Classes of the CIFAR-100 Dataset")
    cols = st.columns(num_columns)

    for i in range(num_columns):
        start_index = i * classes_per_column
        end_index = (i + 1) * classes_per_column if i < num_columns - 1 else num_classes
        
        list_content = ""
        for j, class_name in enumerate(cifar100_classes[start_index:end_index]):
            global_index = start_index + j + 1
            list_content += f"{global_index}. **{class_name.title()}**\n"
        
        cols[i].markdown(list_content)
        
st.subheader('Accuracy Achieved per Model Architecure')

data = {
    'Arquitecture':['LeNet-5', 'AlexNet', 'VGGNet', 'GooLeNet', 'ResNet', 'WideResNet'],
    'Accuracy': [2.54, 99.86, 1, 98.81, 63.22, 81.5],
    'Val Accuracy': [2.41, 43.34, 1, 41.46, 44.36, 65.4]
}

df_arq_acc = pd.DataFrame(data)
st.dataframe(df_arq_acc, use_container_width=False, hide_index=True)

st.markdown("All tested architectures exhibited severe overfitting, evidenced by a significantly large gap between training and validation accuracy. The Wide ResNet architecture, utilizing specific parameter configurations and techniques, resulted in the lowest generalization gap.")

st.header('📸 Upload an image to classify')

uploaded_file = st.file_uploader(
    'Select an image file (JPEG, PNG)',
    type = ['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    try:
        st.success('File succesufly uploaded')
        
        image = Image.open(uploaded_file)
        st.image(image, caption='Input image', use_container_width=True)
        
        st.subheader('Classification Result')

        prediction = make_prediction(image)
        st.subheader(prediction[0])
        st.subheader(cifar100_classes[prediction[0]])
        
    except Exception as e:
        st.error(f'image processing error: {e}')