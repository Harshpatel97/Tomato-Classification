import streamlit as st
from PIL import Image
from prediction import predict, data_transform, class_names
import torch
import warnings
warnings.filterwarnings("ignore")

# Loading the trainer model from models folder
saved_model = torch.load("models\model.pth", 
                         map_location=torch.device('cpu')).to('cpu')

# Heading of our project
st.header('Tomato leaf Disease Classification.', divider='rainbow')

# Upload you leaf image
with st.sidebar:
    imagefile = st.file_uploader("Upload tomato leaf image", type="jpg")

# Process and results
if imagefile is not None:
    st.image(imagefile)

    pred = predict(imagefile)
    st.write("Top three classes with probabilities.")
    st.dataframe(pred)
