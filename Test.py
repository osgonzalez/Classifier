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





print("\n################# Single Variable Resgresion (LSTAT) #################")


#Shuffle and Split Dataset in Train and Trest Datasets

#variables = pd.DataFrame(np.c_[pandasDataframe['LSTAT'], pandasDataframe['RM']], columns = ['LSTAT','RM'])
variables = pd.DataFrame(np.c_[pandasDataframe['LSTAT']], columns = ['LSTAT'])
target = pandasDataframe['target']

variables_train, variables_test, target_train, target_test = train_test_split(variables, target, test_size = 0.2, random_state=5)

print("variables_train (Row,Col) ->", variables_train.shape)
print("variables_test (Row,Col) ->", variables_test.shape)
print("target_train (Row,Col) ->", target_train.shape)
print("target_test (Row,Col) ->", target_test.shape)

# LinearRegression Factory
linearRegression = linear_model.LinearRegression()

#Train the model
linearRegression.fit(variables_train,target_train)

#Predict the result
variablesPredicted = linearRegression.predict(variables_train)

# The Coefficient (m)
print('\nCoefficient (m): ', linearRegression.coef_[0])
# The Independent term (b) (valor que corta el eje en X=0)
print('\nIndependent term (b): ', linearRegression.intercept_)
# Mean squared error
print("\nMean squared error: %.2f" % mean_squared_error(target_train, variablesPredicted))
# Variance score. (The best score is 1.0)
print('Variance score: %.2f' % r2_score(target_train, variablesPredicted))

print("\ny =   mX  +  b \ny = ",linearRegression.coef_[0],"* X +",linearRegression.intercept_);




print("\nThe predicted target of 5 is", linearRegression.predict(np.array([[5]])))
print("The predicted target of 15 is", linearRegression.predict(np.array([[15]])))
print("The predicted target of 20 is", linearRegression.predict(np.array([[20]])))
print("The predicted target of 25 is", linearRegression.predict(np.array([[25]])))

print("\n################# Done #################")











print("\n\n\n################# MultiVariable Resgresion (LSTAT and AGE) #################")


multiVariables = pd.DataFrame(np.c_[pandasDataframe], columns = pandasDataframe.keys())
multiVariables = multiVariables.drop(['target'], axis=1)
target = pandasDataframe['target']

multiVariables_train, multiVariables_test, target_train, target_test = train_test_split(multiVariables, target, test_size = 0.2, random_state=5)

# LinearRegression Factory
regressionMulti = linear_model.LinearRegression()

#Train the model
regressionMulti.fit(multiVariables_train,target_train)

#Predict the result
multiVariablesPredicted = regressionMulti.predict(multiVariables_train)

# The Coefficient (m_sub_N)
print('\nCoefficients (m_sub_N): ', regressionMulti.coef_)
# The Independent term (b) (valor que corta el eje en X=0)
print('\nIndependent term (b): ', regressionMulti.intercept_)
# Mean squared error
print("\nMean squared error: %.2f" % mean_squared_error(target_train, multiVariablesPredicted))
# Variance score. (The best score is 1.0)
print('Variance score: %.2f' % r2_score(target_train, multiVariablesPredicted))

#print("\ny =   m_1 * X  + m_2 * Z +  b \ny = ",regressionMulti.coef_[0],"* X +",regressionMulti.coef_[1],"* Z +",regressionMulti.intercept_);


print("\nThe predicted target with all vars is", regressionMulti.predict(np.array([[632,18,2.31,0,538,7,65.2,4.09,1,296,15.3,396.9,4.98]]))," (The real target is 24)")
print("The predicted target with only LSTAT: 5 is", linearRegression.predict(np.array([[5]])))

#print("\nThe predicted target of LSTAT: 20 and AGE: 94 is", regressionMulti.predict(np.array([[20,94]]))," (The real target is 15)")
#print("The predicted target with only LSTAT: 20 is", linearRegression.predict(np.array([[20]])))


print("\n################# Done #################")

