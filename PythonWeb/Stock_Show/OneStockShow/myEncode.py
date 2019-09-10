# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 15:51:59 2018

@author: Administrator
"""
'''
import tushare as ts

class today_stock():
    today_stock_name=ts.get_today_all().name
'''
#处理json不能讲array序列化的问题:讲array变为list，在views中调用时用到
import json
import numpy as np
  
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8');
        return json.JSONEncoder.default(self, obj)