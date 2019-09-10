# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 09:32:50 2018

@author: Administrator
"""

import OneStockShow.train_model as tm
import OneStockShow.predict_stock as ps
import tensorflow as tf
#from . import models 

def main_predict(stock_num1,have_pre):
    if have_pre==False:
        with tf.variable_scope('train'):
            tm.tr.train_lstm(stock_num1)
        #models.stockModeld.objects.filter(stock_num=stock_num1).update(have_pre=True) 
        #模型预测，将是否预测改为true 
    with tf.variable_scope('train',reuse=True):
        test=ps.prediction(stock_num1)
    return test
main_predict('000777',False)



    