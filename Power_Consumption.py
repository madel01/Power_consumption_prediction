# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 18:38:22 2023

@author: mohamed
"""

############    imports   #################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math 
from sklearn.model_selection import train_test_split
import random
import seaborn as sns
import statsmodels.api as sm


############    Read Data     ############### 
data = pd.read_csv('powerconsumption1.csv')


############   Using datetime as index    ##########

data.index = pd.to_datetime(data['Datetime'])
data = data.drop(['Datetime', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3'], axis=1)

############     Resample every 24 rows into one row    #########

daily_data = data.resample('24h').mean()
#daily_data1 = data.resample('24h').median()


############   Study the correlation between variables    ##########

# Plotting the heatmap to check the correlation among the columns
plt.rcParams["figure.figsize"] = (10,6)
sns.heatmap(daily_data.corr(), annot =True)
sns.heatmap(daily_data.corr(), annot =True)
plt.title('Correlation Matrix')


###########   Study the correlation between each variable and the target variable   ##########

# Plotting Scatter plot of every independent variable with target variable
ax = daily_data.plot.scatter(x='DiffuseFlows', y='PowerConsumption_Zone1', figsize=(4, 4))
ax.set_xlabel('DiffuseFlows')
ax.set_ylabel('PowerConsumption_Zone1')
ax.axhline(0, color='grey', lw=1)
ax.axvline(0, color='grey', lw=1)


###########   Splitting the data to features and labels   ########

X = daily_data.loc[:,["Temperature","Humidity","WindSpeed","GeneralDiffuseFlows","DiffuseFlows"]]
Y = daily_data.loc[:,["PowerConsumption_Zone1"]]

###########   Encoding the categorical variables    ##########

# we don't use it here because we don't have categorical varabile 
# but it can be helpful in such situations 

#X_encoded = pd.get_dummies(data=X, prefix='ENC',prefix_sep='_',
#              columns=['State'],
#              drop_first=True)


##########   Converting to numpy array and shuffling    ########

X = np.array(X)
Y = np.array(Y)

p = np.random.RandomState(seed=42).permutation(X.shape[0])
X = X[p]
Y = Y[p]


##########     Adding Bias feature      #########

X = np.concatenate((np.ones(shape=(364,1)),X),axis=1)


##########     train/test split      #########
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, train_size = .8)


########     Feature Normalization     #######

X1Max = X_train[:,1].max()
X2Max = X_train[:,2].max()
X3Max = X_train[:,3].max()
X4Max = X_train[:,4].max()
X5Max = X_train[:,5].max()

X1Min = X_train[:,1].min()
X2Min = X_train[:,2].min()
X3Min = X_train[:,3].min()
X4Min = X_train[:,4].min()
X5Min = X_train[:,5].min()

X_train[:,1] = (X_train[:,1]-X1Min)/(X1Max-X1Min)
X_train[:,2] = (X_train[:,2]-X2Min)/(X2Max-X2Min)
X_train[:,3] = (X_train[:,3]-X3Min)/(X3Max-X3Min)
X_train[:,4] = (X_train[:,4]-X4Min)/(X4Max-X4Min)
X_train[:,5] = (X_train[:,5]-X5Min)/(X5Max-X5Min)

X_test[:,1] = (X_test[:,1]-X1Min)/(X1Max-X1Min)
X_test[:,2] = (X_test[:,2]-X2Min)/(X2Max-X2Min)
X_test[:,3] = (X_test[:,3]-X3Min)/(X3Max-X3Min)
X_test[:,4] = (X_test[:,4]-X4Min)/(X4Max-X4Min)
X_test[:,5] = (X_test[:,5]-X5Min)/(X5Max-X5Min)

#######################

n = X_train.shape[0]
m = X_test.shape[0]
n_1 = X_train.shape[1]

thetas_original = np.random.rand(n_1,1)


####################################################
####    Ordinary least squares (OLS) method    #####
####################################################

X_opt = X_train[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:,[0,1,2,3,4]]
regressor_OLS = sm.OLS(endog = Y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:,[0,1,3,4]]
regressor_OLS = sm.OLS(endog = Y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:,[0,1,4]]
regressor_OLS = sm.OLS(endog = Y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:,[0,1]]
regressor_OLS = sm.OLS(endog = Y_train, exog = X_opt).fit()
regressor_OLS.summary()


###################################################
####              SGD with momentum           #####
###################################################

def momentum_method(X_train, Y_train, batch_size, learning_rate, momentum_factor, thetas_original):
    
    #thetas = np.zeros((n_1,1))
    n = X_train.shape[0]
    n_1 = X_train.shape[1]
    velocity = 0.0
    thetas = thetas_original
    alpha = learning_rate
    d_thetas = np.zeros((n_1,1))
    Beta = momentum_factor
    
    #delta = 0.00001
    #grad_norm_stop_cond = 0.1
    #convergence_check = 0.001
    
    Error_history = []
    thetasHistory = []
    d_thetasHistory = []
    
    for i in range(60):
        for j in range(0,n,batch_size):
    
            if j+(2*batch_size) > n :
        
               #step1
               h = X_train[j:] @ thetas # prediction vector 

               #step2
               e = h - Y_train[j:] # Error vector 
        
               #step3
               d_thetas = (X_train[j:].T @ e) / (n-j)
        
               #step4
               J = (np.linalg.norm(e)**2) / (2*(n-j)) # Error value
               Error_history.append(J)
        
        
            else:
        
               # step1
               h = X_train[j:j+batch_size] @ thetas # prediction vector 

               #step2
               e = h - Y_train[j:j+batch_size] # Error vector 

               #step3
               d_thetas = (X_train[j:j+batch_size].T @ e) / batch_size
        
               #step4
               J = (np.linalg.norm(e)**2) / (2*batch_size) # Error value
               Error_history.append(J)
    
    
    
            #step5
            velocity = (Beta * velocity) -  (alpha * d_thetas)
            thetas = thetas + velocity
    
            thetasHistory.append(thetas)
            d_thetasHistory.append(d_thetas)
    
            if j+(2*batch_size) > n :
               break
    
    return Error_history,thetas



#############     Hyperparameters    #############
lr = 0.1
momentum_factor = 0.9
batch_size = 16
       

#############     Model training     #############
res_Error,res_thetas = momentum_method(X_train, Y_train, batch_size, lr, 
                                           momentum_factor, thetas_original)

#############     Model testing      #############
pred = X_test @ res_thetas
from sklearn.metrics import r2_score
r2score = r2_score(Y_test,pred)
print("R2 score = ",r2score)


############      Error plotting     #############    
plt.plot(range(0,len(res_Error)),res_Error)







########################################
#               Adam                   #
########################################

def Adam_method(X_train, Y_train, batch_size, learning_rate, thetas_original, m1, m2):
    
    alpha = learning_rate
    Beta1 = m1
    Beta2 = m2
    m = 0.0
    v = 0.0

    epsilon = 0.00001
    t = 0

    n = X_train.shape[0]
    n_1 = X_train.shape[1]

    thetas = thetas_original
    d_thetas = np.zeros((n_1,1))


    #delta = 0.00001
    #thetas = np.zeros((n_1,1))
    #grad_norm_stop_cond = 0.1
    #convergence_check = 0.001
    
    Error_history = []
    thetasHistory = []
    d_thetasHistory = []
    
    
    for i in range(10000):
        for j in range(0,n,batch_size):
     
            t = t + 1
    
            if j+(2*batch_size) > n :
        
               #step1
               h = X_train[j:] @ thetas # prediction vector 

               #step2
               e = h - Y_train[j:] # Error vector 
        
               #step3
               d_thetas = (X_train[j:].T @ e) / (n-j)
        
               #step4
               J = (np.linalg.norm(e)**2) / (2*(n-j)) # Error value
               Error_history.append(J)
        
        
            else:
        
               # step1
               h = X_train[j:j+batch_size] @ thetas # prediction vector 

               #step2
               e = h - Y_train[j:j+batch_size] # Error vector 

               #step3
               d_thetas = (X_train[j:j+batch_size].T @ e) / batch_size
        
               #step4
               J = (np.linalg.norm(e)**2) / (2*batch_size) # Error value
               Error_history.append(J)
    
    
    
            #step5
            m = (Beta1 * m) +  ((1-Beta1) * d_thetas)
            v = (Beta2 * v) +  ((1-Beta2) * pow(d_thetas,2))
    
            m_bar = m / (1-pow(Beta1,t))
            v_bar = v / (1-pow(Beta2,t)) 
    
            thetas = thetas - (alpha * m_bar / (np.sqrt(v_bar)+epsilon))
    
            thetasHistory.append(thetas)
            d_thetasHistory.append(d_thetas)
    
            if j+(2*batch_size) > n :
               break
           
    
    return Error_history,thetas



#############     Hyperparameters    #############
lr = 0.01
m1 = 0.9
m2 = 0.99 
batch_size = 16
       

#############     Model training     #############             
res_Error,res_thetas = Adam_method(X_train, Y_train, batch_size, lr, 
                                                thetas_original, m1, m2)

#############     Model testing      #############
pred = X_test @ res_thetas
from sklearn.metrics import r2_score
r2score = r2_score(Y_test,pred)
print("R2 score = ",r2score)


############      Error plotting     ############# 
plt.plot(range(0,len(res_Error)),res_Error)

