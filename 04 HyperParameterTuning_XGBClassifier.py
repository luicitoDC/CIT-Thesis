###############################################################################
# Hyper-parameter Tuning
###############################################################################

#------------------------------------------------------------------------------
# Read in Required Libraries
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold #, cross_val_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import dask.array as da
from dask_ml.decomposition import PCA
#from dask_ml.model_selection import RandomizedSearchCV

import time
import datetime
import pickle

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
        X = np.load('Data/dataFinal.npy')
        X = pd.DataFrame(data=X[0:,0:])
        X.columns = [str(col) + '_x' for col in X.columns]
        X['index'] = X['43095_x'].astype(int)
        X.drop(['43095_x'], axis=1, inplace=True)
        
    elif transferlearning.upper() == "NO":
        ## Features - With no Features Extraction
        X = np.load('Data/dataFinal1.npy')
        X = pd.DataFrame(data=X[0:,0:])
        X.columns = [str(col) + '_x' for col in X.columns]
        X['index'] = X['67500_x'].astype(int)
        X.drop(['67500_x'], axis=1, inplace=True)
        
    ## Labels
    Y = pd.read_csv('Data/labels2.csv', header = 0)
    
    return (X, Y)

 
def sampleData(XData, YData, n):
    
    ## Features
    X = XData
    
    ## Labels
    Y = YData
    
    ## Merge Dataset
    data = X.merge(Y, on='index', how = 'inner') 
  
    data.drop(['index'], axis=1, inplace=True)
    
    n2 = int(n/8)
    df = data.groupby('Y', group_keys=False).apply(lambda x: x.sample(min(len(x), n2), replace = False))
    df.reset_index(drop=True)
    
    # Final Label
    label = df['Y'].values.ravel()
    
    # Final Features
    df.drop(['Y'], axis=1, inplace=True)
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
    print(" ")
    
    return X

#------------------------------------------------------------------------------
# Hyper-parameter Tuning
#------------------------------------------------------------------------------

def hptuneXGB(model, parameters, X, Y):
    
    modelXGB = model
    
    kFold = StratifiedKFold(n_splits=3, shuffle=True, random_state=1221)
    
    #DASK
#    randomSearch = RandomizedSearchCV(modelXGB, param_distributions = parameters, 
#                                       scoring='accuracy', n_jobs = -1, cv=kFold, #verbose = 3, 
#                                       random_state = 1221, return_train_score = True)
    
    #Scikit
    randomSearch = RandomizedSearchCV(modelXGB, param_distributions = parameters, 
                                      scoring='accuracy', n_jobs = -1, cv=kFold, verbose = 3, 
                                       random_state = 1221, return_train_score = True)
    
    grid_result = randomSearch.fit(X, Y)
    
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
def main(transferlearning = "YES"):         
    # Load Data
    os.chdir('C:/Users/ai/Documents/CITDissertation/Explore')
    XData, YData = loadData(transferlearning)
    
    # Set Directory where outputs will be saved
    if transferlearning == "YES":
        os.chdir('C:/Users/ai/Documents/CITDissertation/Explore/Data/HTXGBoostYesFE')
    else:
        os.chdir('C:/Users/ai/Documents/CITDissertation/Explore/Data/HTXGBoostNoFE')
        
    for n in [200, 400, 1000, 2000, 3000, 4000, 5000]:      
        print("------------------------------------------------------------------")
        print("Analyzing Data with Transfer Learning =  ", transferlearning, " and size of ", n)
        print("------------------------------------------------------------------")
        
        ## Set the start time
        print("Start time:")
        start_time = time.time()
        print(datetime.datetime.now())
            
        # Select the required number of samples
        X, Y = sampleData(XData, YData, n)
        
        # Perform PCA on the Features to Reduce the Dimension
        i = X.shape[0]
        X = preprocessFeatures(X, i) 
        
        # Set aside Test Datasets for Checking on the Training Performance at a later stage
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=1981) 
        
        # Save the Data
        data = [X_train, X_test, Y_train, Y_test]
        fileNameD = "data%s.pickle" % n
        pickle.dump(data, file = open(fileNameD, "wb"))
        
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
        fileNameGR = "grid_result%s.pickle" % n
        pickle.dump(grid_result, file = open(fileNameGR, "wb"))
        
        # Calculate the total amount of time required to compare the baseline models
        print(" ")
        print("Total Time: %s minutes ---" % round((time.time() - start_time)/60,2))
        print(datetime.datetime.now())
        print(" ")
               
        # Reload the file
        #grid_result200 = pickle.load(open("grid_result200.pickle", "rb"))

# First Run
main(transferlearning = "YES")

# Second Run
main(transferlearning = "No")



