###############################################################################
# Hyper-parameter Tuning for Extreme Gradient Boosting
###############################################################################

#------------------------------------------------------------------------------
# Read in Required Libraries
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold #, cross_val_score
from xgboost import XGBClassifier

import dask.array as da
from dask_ml.decomposition import PCA

import time
import datetime
import pickle

import matplotlib
import os
import random

# Set Random Seed
myStudentNum = 181518 
random.seed(myStudentNum)

#------------------------------------------------------------------------------
# Load the Data
#------------------------------------------------------------------------------
def loadData(transferlearning = "YES"):
    ## Features
    if transferlearning.upper() == "YES":
        ## Features - With Features Extraction
        data = np.load("./Data/dataWithTL.npy")
        data = pd.DataFrame(data=data[0:,0:])
        data.columns = [str(col) + '_x' for col in data.columns]
        data['y'] = data['43095_x'].astype(int)
        data.drop(['43095_x'], axis=1, inplace=True)
        
    elif transferlearning.upper() == "NO":
        ## Features - With no Features Extraction
        data = np.load('Data/dataWithoutTL.npy')
        data = pd.DataFrame(data=data[0:,0:])
        data.columns = [str(col) + '_x' for col in data.columns]
        data['y'] = data['67500_x'].astype(int)
        data.drop(['67500_x'], axis=1, inplace=True)
        
    return data

 
def sampleData(data, n):
    
    n2 = int(n/8)
    df = data.groupby('y', group_keys=False).apply(lambda x: x.sample(min(len(x), n2)))
    df.reset_index(drop=True)
    
    # Final Label
    label = df['y'].values.ravel()
    
    # Final Features
    df.drop(['y'], axis=1, inplace=True)
    features = df.values
    
    return (features, label)

#------------------------------------------------------------------------------
# The preprocessFeatures function performs principal components analysis on the
# features that were extracted from the YOLOv3 network. Since the output of the 
# YOLOv3 is a highly dimensional data, the number of features should be reduced
# to at most the number of samples to avoid a serious case of overfitting.
#------------------------------------------------------------------------------
def preprocessFeatures(Features, i):
    print("----")
    print("Extracting the Principal Components of the Features. Please wait...")
    t1 = time.time()
    
    # Prepare Features
    ## Transpose the dataset
    data1 = np.transpose(Features)
    
    ## Performs PCA to reduce the number of Features
    data2 = da.from_array(data1, chunks=data1.shape)
    pca = PCA(n_components = i) #.shape[1]
    pca.fit(data2)
    
    ## Get the Total variance that is explained by the selected Principal Components
    var = 0
    for i in pca.explained_variance_ratio_:
        var += i
    print("Total Variance:")
    print(var) 
    
    ## Print the Principal Component Scores
    #print(pca.singular_values_)
    
    ## Get the Principal Components
    PC = pca.components_
    X = np.transpose(PC)
    
    print(" ")
    print("PCA Duration: %s minutes" % round((time.time() - t1)/60,2))
    
    return X

#--------------------------------------

def hptuneXGB(model, parameters, X, Y):
    
    modelXGB = model
    
    kFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1221)
      
    #Scikit
    searchResult = RandomizedSearchCV(modelXGB, param_distributions = parameters, 
                                      scoring='accuracy', n_jobs = -1, cv=kFold, verbose = 3, 
                                       random_state = 1221, return_train_score = True)
    
#    searchResult = GridSearchCV(modelXGB, param_grid = parameters, 
#                                      scoring='accuracy', n_jobs = -1, cv=kFold, verbose = 3, 
#                                      return_train_score = True)
    
    grid_result = searchResult.fit(X, Y)
    
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    
    Mean_ = grid_result.cv_results_['mean_test_score']
    StdDev = grid_result.cv_results_['std_test_score']
    Params = grid_result.cv_results_['params']
    
    for mean, stdev, param in zip(Mean_, StdDev, Params):
        print("%s (%s) with: %r" % (mean, stdev, param))
    	#print("%f (%f) with: %r" % (mean, stdev, param))
           
    return grid_result


#------------------------------------------------------------------------------
# Putting All the Steps together
#------------------------------------------------------------------------------
def main(transferlearning = "YES", sampleSize=200):         
    # Load Data
    data = loadData(transferlearning = transferlearning)
        
    for n in sampleSize:      
        random.seed(1981)
                    
        print("------------------------------------------------------------------")
        print("Analyzing Data with Transfer Learning =  ", transferlearning, " and size of ", n)
        print("------------------------------------------------------------------")
        
        ## Set the start time
        print("Start time:")
        start_time = time.time()
        print(datetime.datetime.now())
            
        # Select the required number of samples
        X, Y = sampleData(data, n)
        
        # Perform PCA on the Features to Reduce the Dimension
        i = int(n*0.1)
        X = preprocessFeatures(X, i)        
        
#        
        # Create an instance of the XGB Classifier
        modelXGB = XGBClassifier(nthread=-1)
        
        # Set-up the parameter grid
        parameters = {
                'n_estimators' : [200, 400, 600, 800],
                'learning_rate' : [0.001, 0.01, 0.1, 0.2],
                'min_child_weight': [1, 5, 10],
                'subsample': [0.5, 0.75, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': [4, 6, 8, 10]
                }
        
        grid_result = hptuneXGB(modelXGB, parameters, X, Y)

        # Save the file
        fileNameGR = "grid_result%sHT_XGB.pickle" % n
        pickle.dump(grid_result, file = open(fileNameGR, "wb"))
        
        # Calculate the total amount of time required to compare the baseline models
        print(" ")
        print("Total Time: %s minutes ---" % round((time.time() - start_time)/60,2))
        print(datetime.datetime.now())
        print(" ")
               
        # Reload the file
        #grid_result200 = pickle.load(open("grid_result200.pickle", "rb"))

# First Run
N1 = [200, 400, 1000, 2000, 3000, 4000, 5000]
main(transferlearning = "YES", sampleSize = N1)

# Second Run
N2 = [200, 400, 1000, 2000, 3000, 4000, 5000]
main(transferlearning = "No", sampleSize = N2) #




