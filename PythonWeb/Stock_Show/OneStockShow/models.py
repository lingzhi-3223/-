from django.db import models

# Create your models here.生成数据库

class stockModeld(models.Model):
    title = models.CharField(max_length=32,default='Title')
    content = models.TextField(null=True)
    stock_num = models.CharField(max_length=32,default='stock_num')
    have_pre=models.NullBooleanField(default=False)
    
    def __str__(self):
        return self.title
    
class stock_content(models.Model):
    Date = models.CharField(max_length=32,default='date')
    Open = models.CharField(max_length=32,default='open')
    Close=models.CharField(max_length=32,default='close')    
    Low =models.CharField(max_length=32,default='low')
    High =models.CharField(max_length=32,default='high')
    stock_num = models.CharField(max_length=32,default='stock_num')
 