# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 22:54:24 2018

@author: Administrator
"""
#从网上爬取数据
import seaborn as sns
sns.set_style("whitegrid")
from datetime import datetime
import tushare as ts
import pandas as pd

class find():
    def find_stock_by_num(stock_num):
        sns.set_style('whitegrid')
        end=datetime.now()#today()#开始时间结束时间，选取最近一年的数据
        start=datetime(end.year-1,end.month,end.day)
        end=str(end)[0:10] #（2018-08-28）
        start=str(start)[0:10]
        #stock=ts.get_hist_data('000777',start,end)#读取一支股票
        stock=ts.get_hist_data(stock_num,'2017-01-01',end='2018-08-024')#读取一支股票
        stock=stock.sort_index(0)
        
        stock.to_csv('E:/PythonText/PythonWeb/Stock_Show/03.csv') #存储数据
        f=open('E:/PythonText/PythonWeb/Stock_Show/03.csv')
        df=pd.read_csv(f)     #读入股票数据
        data=df.loc[:,['date','open','close','low','high']]  #取第2-6列
        return data

   # find_stock_by_num('000062')