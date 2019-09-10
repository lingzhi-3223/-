# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 15:05:34 2018

@author: Administrator
"""
import numpy as np
#import pandas as pd
import find_stock_by_num as fs
'''
f=open('dataset/300104.csv')
df=pd.read_csv(f)     #读入股票数据
data1=df.loc[:,['high','low','open','close']].values  #取第2-6列
'''

def get_data(stock_num):
    data=fs.find.find_stock_by_num(stock_num)
    data=data.loc[:,['high','low','open','close']].values
    return data

#获取训练集
def get_train_data(stock_num,batch_size=60,time_step=1):
    data=get_data(stock_num)
    batch_index=[]
    data_train=data[:-2]#取所有行
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:4]
       y=normalized_train_data[i+1:i+time_step+1,0,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y

#获取测试集
def get_test_data(stock_num,time_step=1):
    data=get_data(stock_num)
    data_test=data[-1]#只取最后一行
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample
    test_x,test_y=[],[]
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:4]
       y=normalized_test_data[i*time_step+1:(i+1)*time_step+1,0]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:4]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,0]).tolist())
    return mean,std,test_x,test_y
