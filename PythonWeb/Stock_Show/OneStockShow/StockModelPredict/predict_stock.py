# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 19:24:14 2018

@author: admin

利用LSTM预测股票每日最高价

"""
import numpy as np
import tensorflow as tf
import train_model as tm
import get_train_test as gt

def prediction(data):
    X=tf.placeholder(tf.float32, shape=[None,1,4])
    mean,std,test_x,test_y=gt.get_test_data(data)
    with tf.variable_scope("sec_lstm",reuse=True):
        pred,_=tm.trainModel.lstm(X)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('model_save4')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)-1):
            prob=sess.run(pred,feed_dict={X:[test_x[-1]]})
            predict=prob.reshape((-1))
            test_predict.extend(predict)
            test_y=np.array(test_y)*std[0]+mean[0]
            test_predict=np.array(test_predict)*std[0]+mean[0]
            acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差程度
        print("The accuracy of this predict:",acc)
    return acc,test_predict



with tf.variable_scope('train',reuse=True):
        test=prediction('000777') 
print (test)