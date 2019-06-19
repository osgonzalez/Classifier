#import LibsPipDownloader
import scipy
import numpy as np
import matplotlib
import pandas as pd
import statsmodels
import sklearn

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#Load DataSet
from sklearn import datasets
irisDataset = datasets.load_iris()

#Transform <class 'sklearn.utils.Bunch'>  to  <class 'pandas.core.frame.DataFrame'>
irisDataframe = pd.DataFrame(irisDataset.data,columns=irisDataset.feature_names)
#irisDataframe['target'] = pd.Series(irisDataset.target)




# The escaler scales and translates each feature individually such that it is in the given range (Default 0-1)
# oldValue = newValue * (max - min) + min
scaler = MinMaxScaler()
#scaledIrisNumpyArray = scaler.fit_transform(irisDataframe)
scaledIrisNumpyArray = irisDataframe


#Split in test and fit vars
vars_train, vars_test, target_train, target_test = train_test_split(scaledIrisNumpyArray, pd.Series(irisDataset.target), random_state=0)

#hiper-parameters
numberOfNeighbors = 7

#Instanciate the classifier
clasifierKNN = KNeighborsClassifier(numberOfNeighbors)

#clasifierKNN.fit(scaledIrisNumpyArray,irisDataset["target"])
clasifierKNN.fit(vars_train,target_train)

print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(clasifierKNN.score(vars_train,target_train)))

print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(clasifierKNN.score(vars_test,target_test)))


print("\nPrediction of [6.6 ,2.9 ,4.6 ,1.3] (Versicolor)")
predictionValue = clasifierKNN.predict([[6.6 ,2.9 ,4.6 ,1.3]])
print(predictionValue, "->", irisDataset.target_names[predictionValue])

print("\nPrediction of [4.8 ,3.4 ,1.6 ,0.2] (setosa)")
predictionValue = clasifierKNN.predict([[4.8 ,3.4 ,1.6 ,0.2]])
print(predictionValue, "->", irisDataset.target_names[predictionValue])

print("\nPrediction of [6.0 ,2.2 ,5.0 ,1.5] (virginica)")
predictionValue = clasifierKNN.predict([[6.0 ,2.2 ,5.0 ,1.5]])
print(predictionValue, "->", irisDataset.target_names[predictionValue])

