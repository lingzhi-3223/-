"""Stock_Show URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
'''
from django.contrib import admin
from django.urls import path
from django.conf.urls import url

urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'helloworld','OneStockShow.views.hello')
]
'''
#2.0版本配置方式
from django.contrib import admin
from django.urls import path,re_path
from OneStockShow import views
 
urlpatterns = [
    path('admin/', admin.site.urls),
    path('hello/',views.hello,name='hello'),
    path('index/',views.index,name='index'),
    re_path('OneStock/(?P<stock_id>[0-9]+)$',views.stock_page,name='stock_page'),
    path('find/',views.find_stock,name='find'),
    path('echarts/',views.echarts,name='echarts'),
    re_path('StockPredict/(?P<stock_id>[0-9]+)$',views.stock_predict,name='StockPredict'),
    
]
