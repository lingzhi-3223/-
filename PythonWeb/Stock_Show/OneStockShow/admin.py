from django.contrib import admin
from OneStockShow.models import stockModeld

# Register your models here.

class StockAdmin(admin.ModelAdmin):
    list_display=('title','content','stock_num')
    
admin.site.register(stockModeld,StockAdmin)