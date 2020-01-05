模型训练：
预处理：
代码 pretrain.py ；数据集 train_val文件夹 ; 生成：train文件夹
训练：
代码 main.py、densenet3Dfull.py ；数据集 train文件夹 ; 生成：model文件夹

train文件夹中是训练集
先运行pretrain.py进行数据处理，生成4组旋转矩阵

运行main.py即可进行训练
densenet3Dfull.py为DenseNet网络




测试：
代码 test.py ；模型 model文件夹；数据集 test文件夹 ；生成：csv文件

model文件夹中含有四个模型文件
运行test.py文件，默认在同文件夹中生成517021910965predict.csv；
修改test.py第一行datapath，可指定训练集所在文件夹；
修改test.py第二行csvpath，可指定预测值生成文件所在文件夹。
