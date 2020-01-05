datapath='./test'
csvpath=''
from keras.models import load_model
import numpy as np
import os
import pandas as pd
W1=-0.5;W2=0.5;W3=0.5;W4=0.5
def predict(modelpath,datapath=datapath):
	X_test1 = np.ones((117, 1, 32, 32, 32), dtype=np.float32, order='C')
	X_test2 = np.ones((117, 1, 32, 32, 32), dtype=np.float32, order='C')
	X_test3 = np.ones((117, 1, 32, 32, 32), dtype=np.float32, order='C')
	X_test4 = np.ones((117, 1, 32, 32, 32), dtype=np.float32, order='C')
	num=0
	for name in range(1, 584, 1):
		fileme = os.path.join(datapath, 'candidate%s.npz' % name)
		if os.path.exists(fileme):
			#print(fileme)
			with np.load(fileme) as npz:
				voxel = npz['voxel']
				seg = npz['seg']
				voxel1 = voxel[34: 66:1, 34: 66:1, 34: 66:1]
				seg1 = seg[34: 66:1, 34: 66:1, 34: 66:1]
				X_test1[num][0] =voxel1* (seg1 * 0.8 + 0.2) / 255
				######################
				voxel1 = voxel[36: 68:1, 33: 65:1, 32: 64:1]
				seg1 = seg[36: 68:1, 33: 65:1, 32: 64:1]
				for i in range(0, 16, 1):
					voxel1[[(31 - i), i], :, :] = voxel1[[i, (31 - i)], :, :]
					seg1[[(31 - i), i], :, :] = seg1[[i, (31 - i)], :, :]
				X_test2[num][0] = voxel1 * (seg1 * 0.8 + 0.2) / 255
				######################
				voxel1 = voxel[33:65:1, 33:65:1, 33:65:1]
				seg1 = seg[33:65:1, 33:65:1, 33:65:1]
				for i in range(0, 16, 1):
					voxel1[[(31 - i), i], :, :] = voxel1[[i, (31 - i)], :, :]
					seg1[[(31 - i), i], :, :] = seg1[[i, (31 - i)], :, :]
				for i in range(0, 16, 1):
					voxel1[:, :, [(31 - i), i]] = voxel1[:, :, [i, (31 - i)]]
					seg1[:, :, [(31 - i), i]] = seg1[:, :, [i, (31 - i)]]
				X_test3[num][0] = voxel1 * (seg1 * 0.8 + 0.2) / 255
				######################
				voxel1 = voxel[35:67:1, 35:67:1, 35:67:1]
				seg1 = seg[35:67:1, 35:67:1, 35:67:1]
				for i in range(0, 16, 1):
					voxel1[[(31 - i), i], :, :] = voxel1[[i, (31 - i)], :, :]
					seg1[[(31 - i), i], :, :] = seg1[[i, (31 - i)], :, :]
				for i in range(0, 16, 1):
					voxel1[:, :, [(31 - i), i]] = voxel1[:, :, [i, (31 - i)]]
					seg1[:, :, [(31 - i), i]] = seg1[:, :, [i, (31 - i)]]
				for i in range(0, 16, 1):
					voxel1[:, [(31 - i), i], :] = voxel1[:, [i, (31 - i)], :]
					seg1[:, [(31 - i), i], :] = seg1[:, [i, (31 - i)], :]
				X_test4[num][0] = voxel1 * (seg1 * 0.8 + 0.2) / 255
				num = num + 1
	X_test1 = np.moveaxis(X_test1, 1, 4)
	X_test2 = np.moveaxis(X_test2, 1, 4)
	X_test3 = np.moveaxis(X_test3, 1, 4)
	X_test4 = np.moveaxis(X_test4, 1, 4)
	model = load_model(modelpath)
	out1 = model.predict(X_test1)[:,1]
	out2 = model.predict(X_test2)[:,1]
	out3 = model.predict(X_test3)[:,1]
	out4 = model.predict(X_test4)[:,1]

	for i in range(0, 117, 1):
		out1[i] = (out1[i] + out2[i] + out3[i] + out4[i]) / 4
	return out1

model1_predict=predict('./model/1.h5')
model2_predict=predict('./model/2.h5')
model3_predict=predict('./model/3.h5')
model4_predict=predict('./model/4.h5')
for i in range(0, 117, 1):
	model1_predict[i] = W1*model1_predict[i] + W2*model2_predict[i] + W3*model3_predict[i] + W4*model4_predict[i]
print('\ntest loss: ', model1_predict)
center= pd.DataFrame(model1_predict)
center.to_csv(os.path.join(csvpath, '517021910965predict.csv'))

