from pathlib import Path
import torch
import random
from typing import Tuple, Dict
from torchvision import transforms
from prediction import gradio_predict, class_names, data_transform
from timeit import default_timer as timer 
import gradio as gr


# Loading the trainer model from models folder
saved_model = torch.load("models\model.pth", 
                         map_location=torch.device('cpu')).to('cpu')

# Location of the images.
loc = Path("Data\PlantVillage")

## Getting the 3 examples from our data to gradio UI.
test_data_paths = list(loc.glob("*/*.jpg"))
example_list = [[str(filepath)] for filepath in random.sample(test_data_paths, k=3)]

# Create title, description and article strings
title = "Tomato Leaf Disease "
description = "An EfficientNetB2 feature extractor computer vision model to classify images of tomato leaf if they are healthy or infected"

# Create the Gradio demo
demo = gr.Interface(fn=gradio_predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), # what are the outputs?
                            gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs 
                    title=title,
                    examples=example_list,
                    description=description)

# Launch the demo!
demo.launch(debug=False, # print errors locally?
            share=True) # generate a publically shareable URL?

