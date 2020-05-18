# -*- coding: utf-8 -*-
###############################################################################
# Extract Features from the Image Data using the Pre-trained YOLOv3 Network
###############################################################################

import os
import numpy as np
import pandas as pd
import pickle
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


#------------------------------------------------------------------------------
# Load the YOLOv3 Model (created using YOLOv3Model.py)
#------------------------------------------------------------------------------
model = load_model('./Output/modelRGB.h5')
model.summary()

#------------------------------------------------------------------------------
# Define the required input shape for the YOLOv3 model
#------------------------------------------------------------------------------
_width, _height = 416, 416

#------------------------------------------------------------------------------
# Load and Process an image
#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
# Extract Faetures of an Image using the pre-trained YOLOv3 Model
#------------------------------------------------------------------------------
def extractFeatures(image):

    # Define the Filename of the Photo
    filename = image
    
    # Load and process the image
    image, image_w, image_h = processImage(filename, (_width, _height))
    
    prediction = model.predict(image)[0]
    
    prediction = prediction.reshape(prediction.shape[0], -1)
    
    return prediction

#------------------------------------------------------------------------------
# Convert the images to a numpy array
#------------------------------------------------------------------------------
def images2Numpy(image):

    # Define the Filename of the Photo
    filename = image
    
    # Load and process the image
    image, image_w, image_h = processImage(filename, (150, 150))
    
    return image

#------------------------------------------------------------------------------
# Set the Folder Paths
#------------------------------------------------------------------------------
path = "D:/CITDissertation/Input/"

folder1 = "01_TUMOR"
folder2 = "02_STROMA"
folder3 = "03_COMPLEX"
folder4 = "04_LYMPHO"
folder5 = "05_DEBRIS"
folder6 = "06_MUCOSA"
folder7 = "07_ADIPOSE"
folder8 = "08_EMPTY"

#------------------------------------------------------------------------------
# Perform Features Extraction of 5,000 images stored in folder1 - folder8
#------------------------------------------------------------------------------
data = []
label = []
index = 0

for folder in [folder1, folder2, folder3, folder4, folder5, folder6, folder7, folder8]:
    pathIn = path + folder
    for filename in os.listdir(pathIn):
        
        image = os.path.join(pathIn,filename)
        
        extractedFeatures = extractFeatures(image)
        extractedFeatures = np.append(extractedFeatures, index)
        
        data.append(extractedFeatures)
        label.append(folder)
        
    index += 1

dataFinal = np.vstack(data)

#df = pd.DataFrame(data=dataFinal)
#df.columns = [str(col) + '_x' for col in df.columns]
#for i in df.columns:
#    print(i)
#class_counts = df.groupby('43095_x').size() 
#print(class_counts)

#------------------------------------------------------------------------------
# save to npy file
#------------------------------------------------------------------------------
np.save('./Data/dataWithTL.npy', dataFinal)
pickle.dump(label, file = open("./Data/label.pickle", "wb"))

#------------------------------------------------------------------------------
# Convert the 5,000 images stored in folder1 - folder8 into a numpy array
#------------------------------------------------------------------------------
data1 = []
label1 = []
index1 = 0

for folder in [folder1, folder2, folder3, folder4, folder5, folder6, folder7, folder8]:
    pathIn = path + folder
    for filename in os.listdir(pathIn):
        
        image = os.path.join(pathIn,filename)   
        processedImage = images2Numpy(image)
        processedImage = np.append(processedImage, index1)
        
        data1.append(processedImage)
        label1.append(folder)
        
    index1 += 1

dataFinal1 = np.vstack(data1)

#df = pd.DataFrame(data=dataFinal1)
#df.columns = [str(col) + '_x' for col in df.columns]
#for i in df.columns:
#    print(i)
#class_counts = df.groupby('67500_x').size() 
#print(class_counts)

#------------------------------------------------------------------------------
# save to npy file
#------------------------------------------------------------------------------
np.save('./Data/dataWithoutTL.npy', dataFinal1)
pickle.dump(label1, file = open("./Data/label1.pickle", "wb"))


