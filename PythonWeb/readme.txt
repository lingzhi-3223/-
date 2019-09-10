1. 整个项目是搭建在Anaconda3上，Tensorflow深度学习框架上的Python web框架Django2.1。所以需要安装TensorFlow，用pip安装numpy,tushare,Django2.1
2. 运行项目：
1) 打开anaconda prompt命令行窗口，输入activate TensorFlow进入TensorFlow，
2) 用cd进入你的项目文件夹地址。输入Django：Python manage.py runserver，启动服务器。
3) 进入网页首页：127.0.0.1:8000/index/。
4) 再次打开anaconda prompt命令行窗口，输入activate TensorFlow进入TensorFlow
5) 输入tensorboard –logdir=“存放logs的地址”运行Tensorboard 。
6) 在股票数据可视化股票预测链接进入tensorboard ,或者直接输入网址：http://PC-201507201753:6006。
3. OneStockShow 是用于保存项目主要内容
4. Logs是用于存放训练的模型
5. module_demo01存放预测模型要用到的训练模型
6. templates文件夹下存放HTML文件，也就是模板
7. admin.py 是后台管理程序
8. find_stock_by_num.py是用于爬取数据
9. myEcode.py是处理json序列化问题
10. Views.py是模型与模板的桥梁，存取模型调用模板
11. Models.py是处理数据相关所有事务
12. stock_demo01是模型训练和预测tensorboard可视化
13. Stock_Show文件夹下是web 的URL配置以及setting 
14. Static文件夹下是echarts3.0的包
