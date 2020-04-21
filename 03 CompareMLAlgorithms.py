###############################################################################
# Compare Baseline Algorithms Using the YOLOv3 Extracted Features
###############################################################################

#------------------------------------------------------------------------------
# Read in Required Packages
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import StratifiedKFold, cross_val_score #KFold, 
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import time
import datetime
import dask.array as da
from dask_ml.decomposition import PCA
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
    df = data.groupby('Y', group_keys=False).apply(lambda x: x.sample(min(len(x), n2)))
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
    
    return X

#------------------------------------------------------------------------------
# The compareBaseline function calculates the accuracies of different baseline
# machine learning algorithms to identify the top 2 most appropriate classifier
# for the analysis data.
#------------------------------------------------------------------------------
def compareBaseline(X, Y):
    print(" ")
    print("Calculating the accuracies of the baseline models. Please wait...")

    # Set the Baseline Models to Identify the Appropriate Classifier
    accuracy = []
    modelNames = []
    
    baselineModels = []
    baselineModels.append(('CART', DecisionTreeClassifier()))
    baselineModels.append(('KNN', KNeighborsClassifier(n_jobs=-1)))
    baselineModels.append(('LR', LogisticRegression(solver='saga',multi_class = 'multinomial', n_jobs=-1)))
    baselineModels.append(('NB', GaussianNB()))
    baselineModels.append(('RF', RandomForestClassifier(n_jobs=-1)))
    baselineModels.append(('SGD', SGDClassifier(n_jobs=-1)))
    baselineModels.append(('XGB',XGBClassifier(nthread=-1, verbose = True)))
    
    # evaluate each model in turn
    scoring = 'accuracy'
    for name, model in baselineModels:
        print("----")
        t1 = time.time()
        kfold = StratifiedKFold(n_splits=10, random_state=7, shuffle=True)
        cvAccuracy = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=-1)
        accuracy.append(cvAccuracy)
        modelNames.append(name)
        
        msg = "%s: %f (%f)" % (name, cvAccuracy.mean(), cvAccuracy.std())
        print(msg)
        print("Duration: %s minutes" % round((time.time() - t1)/60,2))
        print(" ")       
    
    return accuracy, modelNames


def main(transferLearning = "YES"):
    ## Set the start time
    print("Start time:")
    start_time = time.time()
    print(datetime.datetime.now())
    
    # Load Data
    XData, YData = loadData(transferlearning = transferLearning)

    accuracyList = pd.DataFrame()
    
    for n in [200, 400, 1000, 2000, 3000, 4000, 5000]:      
        
        # Select the required number of samples
        X, Y = sampleData(XData, YData, n)
                 
        # Reduce the number of dimensions of the Features
        i = X.shape[0]
        X = preprocessFeatures(X, i)
            
        # Calculate the Accuracy of each baseline model
        accuracy, models = compareBaseline(X,Y)
        
        # Boxplot of the accuracies of the Baseline Models
        fig = pyplot.figure()
        fig.suptitle('Boxplot: Accuracy of Baseline Classifiers')
        ax = fig.add_subplot(111)
        pyplot.boxplot(accuracy)
        ax.set_xticklabels(models)
        pyplot.show()
        
        # Save the accuracies and model names to disk   
        accuracyDF = pd.DataFrame(accuracy)
        accuracyDF['Model'] = models
        accuracyDF.reset_index(drop=True)
        accuracyDF = accuracyDF.T
        accuracyDF.columns = accuracyDF.iloc[-1]
        accuracyDF = accuracyDF[:-1]
        accuracyDF['NSample'] = n
        
        accuracyList = accuracyList.append(accuracyDF, ignore_index=True)
        file = 'Data/SpotCheckAlgorithms/accuracyList%sFE.txt' % transferLearning
        accuracyList.to_csv(file, sep=',', index=False)
    
    file = 'Data/SpotCheckAlgorithms/accuracyList%sFE2Complete.txt' % transferLearning
    accuracyList.to_csv(file, sep=',', index=False)
    
    # Calculate the total amount of time required to compare the baseline models
    print("Total Time: %s minutes ---" % round((time.time() - start_time)/60,2))
    print(datetime.datetime.now())
    print(" ")
    
    print("Comparison of baseline models has been completed.")
    print("The output files are 'Data/accuracy.txt' and 'Data/modelNames.txt'")


# With Transfer Learning
main(transferLearning = "YES")

# With No Transfer Learning
#main(transferLearning = "NO")



