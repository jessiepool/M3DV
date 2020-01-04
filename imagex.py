"""
import numpy as np
from matplotlib import pyplot as plt
tmp = np.load('../train/train_val/candidate3.npz')
voxel = tmp['voxel']/255
seg = tmp['seg']
for i in range(30,70,1):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(seg[i])
    plt.subplot(1,2,2)
    plt.imshow(voxel[i])
    plt.show()"""

import numpy as np
import os


for name in range(1, 585, 1):
	fileme=os.path.join('./train/train_val', 'candidate%s.npz' % name)
	#fileme = os.path.join('C:/Users/IVYPOOL/Desktop/training/test/test', 'candidate%s.npz' % name)
	if os.path.exists(fileme):
		print(fileme)
		with np.load(fileme) as npz:
			voxel = npz['voxel']
			seg = npz['seg']

			voxel1 = voxel[34: 66:1, 34: 66:1, 34: 66:1]#34 66
			seg1 = seg[34: 66:1, 34: 66:1, 34: 66:1]
			print("1")
			#print(voxel)
			np.savez(
				(os.path.join('./train/train_new1/', 'candidate%s.npz' % name)), voxel=voxel1, seg=seg1)
			print(seg1.shape)
#########################

			voxel1 = voxel[36: 68:1, 33: 65:1, 32: 64:1]  # 34 66
			seg1 = seg[36: 68:1, 33: 65:1, 32: 64:1]

			for i in range (0, 16, 1):
				voxel1[[(31-i),i],:,:]= voxel1[[i,(31-i)],:,:]
				seg1 [[(31-i),i],:,:]= seg1[[i,(31-i)],:,:]
			print("2")
			#print(voxel)
			np.savez(
				(os.path.join('./train/train_new2/', 'candidate%s.npz' % name)), voxel=voxel1, seg=seg1)
			#[36: 68:1, 33: 65:1, 32: 64:1]
######################

			voxel1 = voxel[33:65:1, 33:65:1, 33:65:1]  # 34 66
			seg1 = seg[33:65:1, 33:65:1, 33:65:1]

			for i in range (0, 16, 1):
				voxel1[[(31-i),i],:,:]= voxel1[[i,(31-i)],:,:]
				seg1 [[(31-i),i],:,:]= seg1[[i,(31-i)],:,:]
			for i in range (0, 16, 1):
				voxel1[:, :,[(31 - i), i]] = voxel1[ :, :,[i, (31 - i)]]
				seg1[:, :,[(31 - i), i]]  = seg1[ :, :,[i, (31 - i)]]
			print("3")
			#print(voxel)
			np.savez(
				(os.path.join('./train/train_new3/', 'candidate%s.npz' % name)), voxel=voxel1, seg=seg1)
			#[33:65:1, 33:65:1, 33:65:1]
######################
			voxel1 = voxel[35:67:1, 35:67:1, 35:67:1]  # 34 66
			seg1 = seg[35:67:1, 35:67:1, 35:67:1]

			for i in range (0, 16, 1):
				voxel1[[(31-i),i],:,:]= voxel1[[i,(31-i)],:,:]
				seg1 [[(31-i),i],:,:]= seg1[[i,(31-i)],:,:]
			for i in range (0, 16, 1):
				voxel1[:, :,[(31 - i), i]] = voxel1[ :, :,[i, (31 - i)]]
				seg1[:, :,[(31 - i), i]]  = seg1[ :, :,[i, (31 - i)]]
			for i in range(0, 16, 1):
				voxel1[:,[(31 - i), i], :] = voxel1[:,[i, (31 - i)],  :]
				seg1[:,[(31 - i), i],  :] = seg1[:,[i, (31 - i)],  :]
			print("4")
			#print(voxel)
			np.savez(
				(os.path.join('./train/train_new4/', 'candidate%s.npz' % name)), voxel=voxel1, seg=seg1)
			#[35:67:1, 35:67:1, 35:67:1]



