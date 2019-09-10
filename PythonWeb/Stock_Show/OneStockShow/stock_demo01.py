# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 09:04:58 2018

@author: serlin
"""

import seaborn as sns
from datetime import datetime
import tushare as ts
import pandas as pd

import csv

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

'''
#————————————————————网上获取数据————————————————————

sns.set_style('whitegrid')
end=datetime.now()#设置开始时间和结束时间，选取最近一年的数据
start=datetime(end.year-1,end.month,end.day)
print(end)
print(start)
end=str(end)[0:10]#对时间的格式进行处理
start=str(start)[0:10]
stock=ts.get_hist_data('300104',start,end)#读取一支股票
stock=stock.sort_index(0)  # 将数据按照日期排序
stock.to_csv('D:/learnFiles/2018小学期/作业/stock_project/01.csv') #存储数据

#————————————————————对数据进行处理————————————————————

stock_y = pd.(stock["close"].values) #用close作为标签stock
stock_y = stock_y.drop(stock_y.index[0]) # 删除第一条数据  也就是标签的第一条数据
stock = stock.drop(stock.index[len(stock)-1])
'''

# 删除最后一条数据，为什么删？因为stock_y少了一条数据，所以删掉最后一整行的数据

#——————————定义常量——————————


rnn_unit=10       #hidden layer units隐藏层
input_size=4      #输入
output_size=1     #输出3
lr=0.0006         #学习率
time_step=20      #时间步



f=open('E:/PythonText/PythonWeb/Stock_Show/03.csv')
df=pd.read_csv(f)     #读入股票数据
data=df.loc[:,['high','low','open','close']].values   #取第3-10列
checkpoint_dir = ''

#——————————获取训练集+生成训练集——————————
def get_train_data(batch_size=60,time_step=2,train_begin=0,train_end=100):#设定参数
    batch_index=[]#初始化batch_index
    data_train=data[train_begin:train_end]#定义data_train
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  
#np.mean(data_train,axis=0)是取中值，得到标准化数据 normalized_train_data

    train_x,train_y=[],[]   #训练集x和y初定义
    for i in range(len(normalized_train_data)-time_step):
    #time_steps表示序列本身的长度，normalized_train_data的长度减去time_step（序列本身的长度）
       if i % batch_size==0:#如果batch_size是i的整数倍，再序列里面加入i
           batch_index.append(i)
           
       x=normalized_train_data[i:i+time_step,:4]
       y=normalized_train_data[i+1:i+time_step+1,0,np.newaxis]
       train_x.append(x.tolist())#增加维度
       train_y.append(y.tolist())
       #将矩阵转换为列表，x由于是被增加过维度的，所以实际上它是一个矩阵
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y
 
#——————————获取测试集——————————
def get_test_data(time_step=2,test_begin=100):
    data_test=data[test_begin:]
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


'''
#——————————————————定义神经网络变量——————————————————

#X=tf.placeholder(tf.float32, [None,time_step,input_size])    #每批次输入网络的tensor，输入张量的形状
#Y=tf.placeholder(tf.float32, [None,time_step,output_size])   #每批次tensor对应的标签
#设置共享变量
#with tf.variable_scope('rnn', reuse=True):
#    weights = tf.get_variable("weight")
#    biases = tf.get_variable("biase")

#输入层、输出层权重、偏置
weights={#权重
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }

biases={#偏置
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }
#tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None

'''
#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置

weights={
        'in':tf.Variable(tf.random_normal([input_size,rnn_unit]),name='in_weights'),
        'out':tf.Variable(tf.random_normal([rnn_unit,1]),name='out_weights')
        }
tf.summary.histogram('weights', weights['out'])
    

biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,]),name='in_biases'),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]),name='out_biases')
        }
tf.summary.histogram('biases', biases['out'])

keep_prob = tf.placeholder(tf.float32 , name = 'keep_prob')   #dropout

#——————————————————定义神经网络变量——————————————————
def lstmCell():    #dropout
    #basicLstem单元
    basicLstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    #dropout
   # drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm , output_keep_prob = keep_prob)
    return basicLstm



#——————————————————定义神经网络变量——————————————————
def lstm(X):    
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in'] 
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  
    #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
   #output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    with tf.name_scope('outputs'):
        pred=tf.matmul(output,w_out)+b_out
        tf.summary.histogram('outputs_pred', pred)
    return pred,final_states

#——————————————————定义神经网络变量——————————————————
X=tf.placeholder(tf.float32, [None,time_step,input_size])    #每批次输入网络的tensor，输入张量的形状
Y=tf.placeholder(tf.float32, [None,time_step,output_size])   #每批次tensor对应的标签


#——————————————————训练模型——————————————————

def train_lstm(batch_size=80,time_step=2,train_begin=0,train_end=100):
    with tf.name_scope('inputs'):
        X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
        Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    with tf.variable_scope("sec_lstm"):
        pred,_=lstm(X)
        
    with tf.name_scope('loss'):
        loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
        tf.summary.scalar('loss',loss)
    with tf.name_scope('train_op'):
        train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    #module_file = tf.train.latest_checkpoint()   
    
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()
    with tf.Session() as sess:     
        writer = tf.summary.FileWriter("E:/PythonText/PythonWeb/Stock_Show/OneStockShow/logs",sess.graph)
        #test_writer = tf.summary.FileWriter("E:/TensorBoard/stock_predict01",sess.graph)  
        sess.run(init)
        #saver.restore(sess, module_file)
        #重复训练2000次
        for i in range(2000):
            for step in range(len(batch_index)-1):
                #_batch_size = 128
                #batch_x, batch_y = mnist.train.next_batch(_batch_size)
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
                sess.run(train_op,feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i,loss_)
            if i % 200==0:
                print("保存模型：",saver.save(sess,'module_demo01/stock_predict.ckpt'))
                result = sess.run(merged,feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
                #test_result = sess.run(merged,feed_dict={X:test_x[batch_index[step]:batch_index[step+1]],Y:test_y[batch_index[step]:batch_index[step+1]]})
                writer.add_summary(result,i)
                #test_writer.add_summary(test_result,i)
    #loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    #loss函数
    #train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    #saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)

    #module_file = tf.train.latest_checkpoint()   
    '''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, module_file)
        #重复训练2000次
        for i in range(500):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
                print("Number of iterations:",i," loss:",loss_)
        print("model_save: ",saver.save(sess,'module_demo01\\modle.ckpt'))
        #I run the code on windows 10,so use  'model_save2\\modle.ckpt'
        #if you run it on Linux,please use  'model_save2/modle.ckpt'
        print("The train has finished")
        '''
#      print(i,loss_)
#    if i % 200==0:
#print("保存模型：",saver.save(sess,'./stock_predict.ckpt',global_step=i))


#with tf.variable_scope('train1'):
train_lstm()

 
#————————————————预测模型————————————————————
def prediction(time_step=2):

    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    with tf.variable_scope("sec_lstm",reuse=True):
        pred,_=lstm(X)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('module_demo01')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})
          predict=prob.reshape((-1))
          test_predict.extend(predict)

        #存储数据
        test_y=np.array(test_y)*std[0]+mean[0]
        test_predict=np.array(test_predict)*std[0]+mean[0]
        
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差程度
        print("The accuracy of this predict:",acc)
        '''
        data1 = pd.DataFrame(test_predict)
        data1.to_csv('04.csv')
        '''
        #以折线图表示结果
    plt.figure()
    plt.plot(list(range(len(test_predict))), test_predict, color='b',)
    plt.plot(list(range(len(test_y))), test_y,  color='r')
    plt.show()
 

prediction()
