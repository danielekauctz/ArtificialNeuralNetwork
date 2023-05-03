"""
Artificial Neural Network (ANN)
   using sklearn.neural_network.MLPRegressor

Example by Daniele Kauctz Monteiro (2023)
danielekauctz@hotmail.com

"""
# Parameters:
# data: training dataset
# X: input (initial data for the neural network)
# y: output (result for given inputs)
# model: Artificial Neural Network

##  REQUIRED MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy import stats

## TRAINING DATA
data = pd.read_csv('data.txt',delim_whitespace=True, header = None,dtype = 'float')
data = data.values

X   = np.array([data[:,0], data[:,2]]).T
y   = data[:,1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, train_size=0.9, random_state=1) # creating training and test dataset

## NEURAL NETWORK
model = MLPRegressor(hidden_layer_sizes=(5,),random_state=1, max_iter=1300)     # network parameters 
model.fit(X_train, y_train)

## RESULTS ANALYSIS
expected_y = y_train
predicted_y = model.predict(X_train)

# expected_y = y_test
# predicted_y = model.predict(X_test)

ind = np.argsort(expected_y)      # sorting data in ascending order
expected_y = expected_y[ind]
predicted_y = predicted_y[ind]

mean_squared_error = metrics.mean_squared_error(expected_y, predicted_y)

## PLOT REGRESSION
t_o = np.arange(0, 11, 1)                             # expected y = predicted y
res = stats.linregress(expected_y, predicted_y)       # fitted line

plt.figure(figsize=(5, 5))
plt.plot(expected_y, predicted_y, 'ko', label ='Data')
plt.plot(expected_y, res.intercept + res.slope*expected_y, 'b', label='Fitted line')
plt.plot(t_o, 'k--', label ='Y = T')
plt.suptitle("Training regression")
plt.xlabel('Target') 
plt.ylabel('Output') 
plt.xlim(0,10),plt.ylim(0,10)
plt.legend(loc = 'lower right')

## PLOT LOSS CURVE
plt.figure(figsize=(9, 5))
plt.plot(model.loss_curve_, 'k')
plt.suptitle("Activation function: ReLU")
plt.xlabel('Iteration')
plt.ylabel('Loss')

## PREDICTION TEST
X_predict = np.array([0.5758, 0.04])
X_predict = X_predict.reshape(-1, 2)
y_predict = model.predict(X_predict)    # true y = 0.5838