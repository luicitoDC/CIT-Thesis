###############################################################################
# Get the Optimal PCA
###############################################################################

#------------------------------------------------------------------------------
# Read in Required Packages
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
import dask.array as da
from dask_ml.decomposition import PCA
import random

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

#class_counts = data.groupby('y').size() 
#print(class_counts)

 
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

#transferLearning = "YES"

def getOptimalPCA(transferLearning):
    data = loadData(transferlearning = transferLearning)
        
    for n in [200, 400, 1000, 2000, 3000, 4000, 5000]:  
        # Set Random Seed
        random.seed(1981)
        
        X, Y = sampleData(data, n)
        
        # Prepare Features
        
        ## Transpose the dataset
        data1 = np.transpose(X)
        
        ## Performs PCA to reduce the number of Features
        data2 = da.from_array(data1, chunks=data1.shape)
        pca = PCA(n_components = n) #.shape[1]
        pca.fit(data2)
        
        # Plot the Cumulative Explained Variance
        print("Transfer Learning = %s" % transferLearning)
        
        fig = pyplot.figure()
        plotTitle = "Elbow Method for Data Size of %s" % n
        fig.suptitle(plotTitle)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Explained Variance")
        pyplot.show()
        

getOptimalPCA("YES")    
getOptimalPCA("No")  
                 

        
        
        
        
        
        
        
        
        