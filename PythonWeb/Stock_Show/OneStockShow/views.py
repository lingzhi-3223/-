from django.shortcuts import render
from django.http import HttpResponse
from . import models 
import OneStockShow.find_stock_by_num as fd
import json
import OneStockShow.myEncode as ed
#import OneStockShow.main_predict as mp
# Create your views here.

def hello(request):
    return HttpResponse('<html>hello world<html>')

def index(request):
    stocks = models.stockModeld.objects.all()
    #allstock=als.today_stock.today_stock_name
    return render(request,'index.html',{'stocks':stocks })
    
def stock_page(request,stock_id):
    stock=models.stockModeld.objects.get(pk=stock_id)
    result=fd.find.find_stock_by_num(stock.stock_num).values
    #print (stock.stock_num)
    return render(request, 'OneStock.html' ,{'stock':stock,'result_json' : json.dumps(result,cls=ed.MyEncoder)})

def find_stock(request):
    find=request.POST.get('find','天天股票')
    #come=models.stockModeld.objects.filter(title__contains=find)   
    stock=models.stockModeld.objects.get(title__contains=find)
    print ("这个数是",find) 
    
    return render(request, 'OneStock.html' ,{'stock':stock})

def echarts(request): 
    #result=fd.find.find_stock_by_num('000777')
    return render(request,'echarts.html')

def stock_predict(request,stock_id): 
    stock=models.stockModeld.objects.get(pk=stock_id)
    #{'stockpredict' : json.dumps(stockpredict,cls=ed.MyEncoder)}
    #stockpredict=mp.m_p.main_predict(stock.stock_num,stock.have_pre),{'stockpredict':stockpredict}}
    return render(request,'StockPredict.html',{'stock',stock})
