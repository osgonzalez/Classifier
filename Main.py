#import LibsPipDownloader
import matplotlib.pyplot as plt
import scipy
import numpy as np
import matplotlib
import pandas as pd
import statsmodels
import sklearn

#Load DataSet
from sklearn.datasets import load_boston
boston_dataset = load_boston()

from sklearn.model_selection import train_test_split

#Load Linear model y MSE (Error Cuadratico Medio)
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

##########################################
#print(boston_dataset.keys())
#print(boston_dataset['feature_names'])
#print(boston_dataset['data'])
#print(boston_dataset['target'])
##########################################


#Transform <class 'sklearn.utils.Bunch'>  to  <class 'pandas.core.frame.DataFrame'>
pandasDataframe = pd.DataFrame(boston_dataset.data,columns=boston_dataset.feature_names)
pandasDataframe['target'] = pd.Series(boston_dataset.target)


#print(pandasDataframe.head())
#print(pandasDataframe.describe())

print("Rows and Colums ->", pandasDataframe.shape)



print("\n################# Check Null Values #################")
print(pandasDataframe.isnull().sum())
print("#####################################################\n")







#Shuffle and Split Dataset in Train and Trest Datasets

variables = pd.DataFrame(np.c_[pandasDataframe['LSTAT'], pandasDataframe['RM']], columns = ['LSTAT','RM'])
target = pandasDataframe['target']

variables_train, variables_test, target_train, target_test = train_test_split(variables, target, test_size = 0.2, random_state=5)

print("variables_train (Row,Col) ->", variables_train.shape)
print("variables_test (Row,Col) ->", variables_test.shape)
print("target_train (Row,Col) ->", target_train.shape)
print("target_test (Row,Col) ->", target_test.shape)





print("Done!")
