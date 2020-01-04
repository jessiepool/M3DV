训练：
train文件夹中是训练集
先运行pretrain.py进行数据处理，生成4组不同方向上旋转的
运行main.py即可进行训练
densenet3Dfull.py为DenseNet网络

测试：
model文件夹中含有四个模型文件，test文件夹中是测试集
运行test.py文件，默认在同文件夹中生成517021910965predict.csv；
修改test.py第一行datapath，可指定训练集所在文件夹；
修改test.py第二行csvpath，可指定预测值生成文件所在文件夹。