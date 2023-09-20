## Overview:
This project is a plant disease classifier developed using the Plant Village dataset. It leverages various data transformation techniques implemented in PyTorch to enhance model performance. The model used for this project is the state-of-the-art EfficientNetB2, which has been fine-tuned and trained on the dataset to achieve an impressive accuracy of 90.4%.



## Upload Your Custom Tomato Leaf Image or Select any one from below and Click on Sumbit or Simply Drag and drop the image
![Screenshot 2023-09-04 182736](https://github.com/Harshpatel97/Tomato-Classification/assets/129877052/909850f2-2984-4348-962c-66a226715335)

## This will show the predictions of model with top three probabilities from classes
![Screenshot 2023-09-04 182747](https://github.com/Harshpatel97/Tomato-Classification/assets/129877052/466e9b88-5a55-4518-8cf9-b701e32535b4)

# How to run?
### STEPS:
 
Clone the repository

```bash
https://github.com/Harshpatel97/Tomato-Classification.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create --name tomato python=3.9 -y
```

```bash
conda activate tomato
```
### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

### STEP 3- Create a Data folder and download the data

https://drive.google.com/file/d/1rneHpIpV0pwDhV27mh8R2caJ9Yx6WTxZ/view?usp=sharing

### STEP 4- Create a models folder and download the pytorch model

https://drive.google.com/file/d/1t9G2HjXHZbM_VHQvhFYfrz5T7nIxnTmP/view?usp=sharing


## Make sure to provide a model.pth(Path) to saved_model variable in app.py

```bash
# Finally run the following command
python app.py
```


