import densenet3Dfull
import os
import numpy as np
np.random.seed(1330)  # for reproducibility
import keras
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split

#################################################
def get_batch(x, y, step, batch_size, alpha=0.2):
    """
    get batch data
    :param x: training data
    :param y: one-hot label
    :param step: step
    :param batch_size: batch size
    :param alpha: hyper-parameter α, default as 0.2
    :return:
    """
    candidates_data, candidates_label = x, y
    offset = (step * batch_size) % (candidates_data.shape[0] - batch_size)

    # get batch data
    train_features_batch = candidates_data[offset:(offset + batch_size)]
    train_labels_batch = candidates_label[offset:(offset + batch_size)]

    # 最原始的训练方式
    if alpha == 0:
        return train_features_batch, train_labels_batch
    # mixup增强后的训练方式
    if alpha > 0:
        weight = np.random.beta(alpha, alpha, batch_size)
        x_weight = weight.reshape(batch_size, 1, 1, 1)
        y_weight = weight.reshape(batch_size, 1)
        index = np.random.permutation(batch_size)
        x1, x2 = train_features_batch, train_features_batch[index]
        x = x1 * x_weight + x2 * (1 - x_weight)
        y1, y2 = train_labels_batch, train_labels_batch[index]
        y = y1 * y_weight + y2 * (1 - y_weight)
        return x, y
#################################################







class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

X=np.ones((465*4,1,32,32,32), dtype=np.float32, order='C')
num=0
for name in range(1, 584, 1):
    fileme=os.path.join('C:/Users/IVYPOOL/Desktop/training/train/train_new', 'candidate%s.npz' % name)
    if os.path.exists(fileme):
        #print(fileme)
        with np.load(fileme) as npz:
            voxel = npz['voxel']
            seg = npz['seg']
            X[num][0]= npz['voxel'] * (npz['seg'] * 0.9 + 0.1) /255      ############改占比
            ###############CT扫描0-255    分割0-1
            #X[num][0] = npz['voxel']  / 255
            num=num+1
######添加旋转
for name in range(1, 584, 1):
    fileme=os.path.join('C:/Users/IVYPOOL/Desktop/training/train/train_new1', 'candidate%s.npz' % name)
    if os.path.exists(fileme):
        #print(fileme)
        with np.load(fileme) as npz:
            voxel = npz['voxel']
            seg = npz['seg']
            X[num][0]= npz['voxel'] * (npz['seg'] * 0.9 + 0.1) /255      ############改占比
            ###############CT扫描0-255    分割0-1
            #X[num][0] = npz['voxel']  / 255
            num=num+1
for name in range(1, 584, 1):
    fileme=os.path.join('C:/Users/IVYPOOL/Desktop/training/train/train_new2', 'candidate%s.npz' % name)
    if os.path.exists(fileme):
        #print(fileme)
        with np.load(fileme) as npz:
            voxel = npz['voxel']
            seg = npz['seg']
            X[num][0]= npz['voxel'] * (npz['seg'] * 0.9 + 0.1) /255      ############改占比
            ###############CT扫描0-255    分割0-1
            #X[num][0] = npz['voxel']  / 255
            num=num+1
for name in range(1, 584, 1):
    fileme=os.path.join('C:/Users/IVYPOOL/Desktop/training/train/train_new3', 'candidate%s.npz' % name)
    if os.path.exists(fileme):
        #print(fileme)
        print(num)
        with np.load(fileme) as npz:
            voxel = npz['voxel']
            seg = npz['seg']
            X[num][0]= npz['voxel'] * (npz['seg'] * 0.9 + 0.1) /255      ############改占比
            ###############CT扫描0-255    分割0-1
            #X[num][0] = npz['voxel']  / 255
            num=num+1

X = np.moveaxis(X, 1, 4)  # 将shape改为(465,32,32,32,1)

sheet = pd.read_csv('C:/Users/IVYPOOL/Desktop/training/train/train_val.csv')
cell_data = np.array(sheet)
Y = np.zeros((465*4,2), dtype = int)
for i in range (0,465*4):
     if cell_data[i][1]==0:
        Y[i][0]=1
     else:Y[i][1]=1
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=7)

#print(train_data[0].size(0))
#print(train_label[0].size(0))

model = keras.models.Sequential()
print("Model created")

model.add(densenet3Dfull.DenseNet((32, 32, 32, 1), classes=2, depth=10, nb_dense_block=3,
                          growth_rate=6, nb_filter=32,dropout_rate=0, weights=None))

print("Finished compiling")
print("Building model...")
model.summary()

adam = Adam(lr=1e-2)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('Training ------------')
history = LossHistory()
model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=20,
          batch_size=40,shuffle = True,callbacks=[history])

history.loss_plot('epoch')
model.save ('./model/try1.h5')


