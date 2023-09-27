import torch
from torchvision import transforms
from PIL import Image
from typing import Tuple, List, Dict
from timeit import default_timer as timer

# Loading the trainer model from models folder
saved_model = torch.load("models\model.pth", 
                         map_location=torch.device('cpu')).to('cpu')


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

data_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor()
])

def predict(image_path):
      
    image = Image.open(image_path)
    
    # Transform the target image and add a batch dimension
    img = data_transform(image).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    saved_model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(saved_model(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class.
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    Predictions = dict(sorted(pred_labels_and_probs.items(), key=lambda item: item[1], reverse=True)[:3])
    return Predictions

def gradio_predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = data_transform(img).unsqueeze(0)
    
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