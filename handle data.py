#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 12:33:06 2018

@author: luohaozheng
"""

import numpy as np
import pandas as pd
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from mlens.ensemble import SuperLearner
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import accuracy_score

def open_file(file):
    try:                              
        df=pd.read_csv(file)        
    except FileNotFoundError:                
        print("***file open error")
    return df

def sort(df,a):
     df=df.sort_values(by=a,ascending=False)
     df1=df[0:20]
     df2=df[-20:]
     return df1,df2


def get_real(data,data2):
    a=data2['uniqcode'].unique()
    for i in range(data.shape[0]):
         b=data[i:i+1]
         if str(b['uniqcode']) in a:
             b['real_choose']=1
         else:
             b['real_choose']=0
    return data

def get_pred(data,data2):
    a=data2['uniqcode'].unique()
    for i in range(data.shape[0]):
         b=data[i:i+1]
         if str(b['uniqcode']) in a:
             b['pred_choose']=1
         else:
             b['pred_choose']=0
    return data

def BayesianRidge(data,data2,data5,during):
    y = data2['prmom'+during+'_f']
    x = data2.drop(['prmom1d_f','prmom1w_f','prmom2w_f','prmom3w_f','uniqcode','date'],axis=1)
    x=x.fillna(0)
    y=np.array(y)
    x=np.array(x)
    reg = linear_model.BayesianRidge()
    reg.fit(x, y)
    X= data5.drop(['prmom1d_f','prmom1w_f','prmom2w_f','prmom3w_f','uniqcode','date','pred'],axis=1)
    X=X.fillna(0)
    X=np.array(X)
    pred1=reg.predict(X)
    data['pred_bay']=pred1
    return data

def deep_learning(data,data2,data5,during):
    y = data2['prmom'+during+'_f']
    x = data2.drop(['prmom1d_f','prmom1w_f','prmom2w_f','prmom3w_f','uniqcode','date'],axis=1)
    x=x.fillna(0)
    y=np.array(y)
    x=np.array(x)
    reg = MLPRegressor(hidden_layer_sizes=(10,),  activation='relu', solver='adam',    alpha=0.001,batch_size='auto',
               learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
               random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
               epsilon=1e-08)
    reg.fit(x, y)
    X= data5.drop(['prmom1d_f','prmom1w_f','prmom2w_f','prmom3w_f','uniqcode','date','pred'],axis=1)
    X=X.fillna(0)
    X=np.array(X)
    pred1=reg.predict(X)
    data['pred_deep']=pred1
    return data

def SVC_pre(data,data2,data5,during):
    y = data2['prmom'+during+'_f']
    x = data2.drop(['prmom1d_f','prmom1w_f','prmom2w_f','prmom3w_f','uniqcode','date'],axis=1)
    x=x.fillna(0)
    y=np.array(y)
    x=np.array(x)
    reg = SVC()
    reg.fit(x, y)
    X= data5.drop(['prmom1d_f','prmom1w_f','prmom2w_f','prmom3w_f','uniqcode','date','pred'],axis=1)
    X=X.fillna(0)
    X=np.array(X)
    pred1=reg.predict(X)
    data['pred_svc']=pred1
    return data

def adaboost_pre(data,data2,data5,during):
    y = data2['prmom'+during+'_f']
    x = data2.drop(['prmom1d_f','prmom1w_f','prmom2w_f','prmom3w_f','uniqcode','date'],axis=1)
    x=x.fillna(0)
    y=np.array(y)
    x=np.array(x)
    reg =  AdaBoostClassifier(n_estimators=100)
    reg.fit(x, y)
    X= data5.drop(['prmom1d_f','prmom1w_f','prmom2w_f','prmom3w_f','uniqcode','date','pred'],axis=1)
    X=X.fillna(0)
    X=np.array(X)
    pred1=reg.predict(X)
    data['pred_adaboost']=pred1
    return data

def esemble(data,data2,data5,during):
    ensemble = SuperLearner(scorer=accuracy_score, random_state=45, verbose=2)
    ensemble.add(linear_model.LinearRegression())
    ensemble.add_meta([GaussianProcessRegressor()])
    y = data2['prmom'+during+'_f']
    x = data2.drop(['prmom1d_f','prmom1w_f','prmom2w_f','prmom3w_f','uniqcode','date'],axis=1)
    x=x.fillna(0)
    y=np.array(y)
    x=np.array(x)
    ensemble.fit(x,y)
    X= data5.drop(['prmom1d_f','prmom1w_f','prmom2w_f','prmom3w_f','uniqcode','date','pred'],axis=1)
    X=X.fillna(0)
    X=np.array(X)
    preds = ensemble.predict(X)
    data['pred_essemble']=preds
    return data

def pre_handle(data2,data5,during):
    train_y=data2['prmom'+during+'_f']
    train=data2.drop(['prmom1d_f','prmom1w_f','prmom2w_f','prmom3w_f','uniqcode','date'], axis=1)
    test=data5.drop(['prmom1d_f','prmom1w_f','prmom2w_f','prmom3w_f','uniqcode','date','pred'], axis=1)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    mean1 = train.mean(axis=0)
    std1 = train.std(axis=0)
    train = (train - mean) / std
    test = (test - mean1) / std1
    train_y=pd.DataFrame(train_y)
    train_y=np.array(train_y)
    scalarY = MinMaxScaler()
    scalarY.fit(train_y.reshape(train_y.shape[0],1))
    train_y = scalarY.transform(train_y.reshape(train_y.shape[0],1))
    return train,train_y,test

def make_model_tensflow(train):
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, 
                       input_shape=(train.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1,activation='linear')])


    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,metrics=['mae'])
    return model

def wider_model():
	# create model
	model = keras.Sequential()
	model.add( keras.layers.Dense(20, input_dim=15, kernel_initializer='normal', activation='relu'))
	model.add( keras.layers.Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
 
def data_predict_tens(data,data2,data5,during):
    (train,train_y,test)=pre_handle(data2,data5,during)
    model=make_model_tensflow(train)
    model.summary()
    model.fit(train, train_y, epochs=1000, verbose=0)
    pred= model.predict(test)
    data['tens_pred']=pred
    return data

def data_predict_keras(data,data2,data5,during):
    (train,train_y,test)=pre_handle(data2,data5,during)
    model= wider_model()
    model.fit(train, train_y, epochs=1000, verbose=0)
    pred= model.predict(test)
    data['keras_pred']=pred
    return data
    

def handle_data(data,during):
    A=data
    A=A.drop(['uniqcode','date'], axis=1)
    B=A.corrwith(A['prmom'+during+'_f'])
    B=abs(B)
    B=B.drop('prmom2w_f',axis=0)
    data['min']=A.min(axis=1)
    data['max']=A.max(axis=1)
    data['Average']=A.mean(axis=1)
    data['std']=A.std(axis=1)
    data['median']=A.median(axis=1)
    data['mad']=A.mad(axis=0)
    data['kurt']=A.kurt(axis=0) 
    data['skew']=A.skew(axis=0)
    data['date']=pd.to_datetime(data['date'])
    a=data['date'].unique()
    for i in range(len(a)):
        C=data[data['date']==a[i]]       
        (df1,df2)=sort(C,['prmom'+during+'_f'])
        (df3,df4)=sort(C,['pred'])
        data3=pd.concat([df1, df2])
        data4=pd.concat([df3,df4])
        C=get_real(C,data3)
        C=get_pred(C,data4)
        data[data['date']==a[i]]=C
    data=BayesianRidge(data,data2,data5,during)
    data=deep_learning(data,data2,data5,during)
    #data=SVC_pre(data,data2,data5,during)
    #data=adaboost_pre(data,data2,data5,during)
    data=data_predict_tens(data,data2,data5,during)
    data=data_predict_keras(data,data2,data5,during)
    data=esemble(data,data2,data5,during)
    return data
    

file=input('file name:')
period=input('period:')
data=open_file(file)
data5=copy.deepcopy(data)
data2= open_file('data_2.csv')   
handle_data(data,period)
