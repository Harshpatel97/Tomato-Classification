import torchvision
from torch import nn
from pathlib import Path
import time
import torch, torchvision

import torch
import random
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from torch import nn

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from timeit import default_timer as timer 
import gradio as gr


loc = Path("E:\Tomato-Classification\Data\PlantVillage")

data_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['Bacterial_spot',
                 'Curl_Virus',
                 'Early_blight',
                 'Healthy',
                 'Late_blight',
                 'Leaf_Mold',
                 'Mosaic_virus',
                 'Septoria_leaf_spot',
                 'Target_Spot',
                 'Two_spotted_spider_mite']

model = torchvision.models.efficientnet_b2().cpu

## Provide the model.pth path from your models folder..
saved_model = torch.load(r"E:\Tomato-Classification\models\model.pth", map_location=torch.device('cpu'))


def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = data_transform(img).unsqueeze(0).to('cpu')
    
    # Put model into evaluation mode and turn on inference mode
    saved_model.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(saved_model(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

test_data_paths = list(loc.glob("*/*.jpg"))
example_list = [[str(filepath)] for filepath in random.sample(test_data_paths, k=3)]
example_list

# Create title, description and article strings
title = "Tomato Leaf Disease "
description = "An EfficientNetB2 feature extractor computer vision model to classify images of tomato leaf if they are healthy or infected"

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    examples=example_list, 
                    title=title,
                    description=description)

# Launch the demo!
demo.launch(debug=False, # print errors locally?
            share=True) # generate a publically shareable URL?

