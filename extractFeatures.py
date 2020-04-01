# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 00:38:02 2020

@author: ai
"""

import os
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model



# Load the YOLOv3 Model (created using YOLOv3Model.py)
model = load_model('model.h5')

# Define the required input shape for the YOLOv3 model
_width, _height = 416, 416

# Load and Process an image
def processImage(file, size):
    
    # Load the image
    image = load_img(file)
    
    # Get the shape of the image
    w, h = image.size
    
    # Load the Image with the target size
    image = load_img(file, target_size = size)
    
    # Convert the Image to a Numpy Array
    image = img_to_array(image)
    
    # Normalize the Values by scaling it to [0, 1])
    image = image.astype('float32')
    image /= 255.0
    
    # Add a dimension to output one sample
    image = expand_dims(image, 0)
    
    # Return the result
    return image, w, h


# Extract Faetures of an Image using the pre-trained YOLOv3 Model
def extractFeatures(image):

    # Define the Filename of the Photo
    filename = image
    
    # Load and process the image
    image, image_w, image_h = processImage(filename, (_width, _height))
    
    prediction = model.predict(image)[0]
    
    prediction = prediction.reshape(prediction.shape[0], -1)
    
    return prediction


folder1 = "C:/Users/ai/Downloads/colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/01_TUMOR"
folder2 = "C:/Users/ai/Downloads/colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/02_STROMA"
folder3 = "C:/Users/ai/Downloads/colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/03_COMPLEX"
folder4 = "C:/Users/ai/Downloads/colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/04_LYMPHO"
folder5 = "C:/Users/ai/Downloads/colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/05_DEBRIS"
folder6 = "C:/Users/ai/Downloads/colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/06_MUCOSA"
folder7 = "C:/Users/ai/Downloads/colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/07_ADIPOSE"
folder8 = "C:/Users/ai/Downloads/colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/08_EMPTY"


# Perform Features Extraction of 5,000 images stored in folder1 - folder8
data = []

for folder in [folder1, folder2, folder3, folder4, folder5, folder6, folder7, folder8]:
    for filename in os.listdir(folder):
    #
        image = os.path.join(folder,filename)
        
        extractedFeatures = extractFeatures(image)
        
        data.append(extractedFeatures)

#