##################################################################################
# Function to predict pH given the model, the folder and the sequence of channels used 
##################################################################################

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from scipy import ndimage
import numpy as np
import os
from scipy import stats#.linregress
import pickle
#from PIL import Image
import cv2
import LoadImagFiles
import sys

is_three = False
if (sys.version_info > (3, 0)):
	is_three = True

def Predict(clf,pca, sc, folder_in, folder_out, position, folder_correct, k, sequence,ch = ["a","b","c","d"]):
	channels = ["a","b","c","d"]
	img=mpimg.imread(folder_in+"a"+"_"+str(k)+".tif")
	S = np.shape(img)

	IM = []
	xx, yy = np.meshgrid(np.arange(S[0]), np.arange(S[1]))
	IM.append(xx.ravel())
	IM.append(yy.ravel())
		
	# Add wavelength as input variables
	for c in channels:
		if c in ch:##
			file = folder_in+c+"_"+str(k)+".tif"
			 
			im = LoadImagFiles.Load(file,folder_correct, c)
			IM.append(im.ravel())#(mpimg.imread(file)*CORRECT).ravel())
		else:
			file = folder_in+ch[0]+"_"+str(k)+".tif"
			IM.append(LoadImagFiles.Load(file, folder_correct, c).ravel())


	X = np.array(IM).transpose()

	# transform the data
	Xp = X[:,sequence]
	Xp  = pca.transform(Xp)						# PCA
	Xp2 = sc.transform(Xp)						# Normalise
	
	X_pred = Xp2[:,0]*0.
	length = len(Xp2)
	sub = 10.
	for i in range(int(sub)):
		if i<int(sub) - 1:
			X_pred[int(i*length/sub):int((i+1)*length/sub)] = clf.predict(Xp2[int(i*length/sub):int((i+1)*length/sub),:])
		else:
			X_pred[int(i*length/sub):-1] = clf.predict(Xp2[int(i*length/sub):-1,:])		
	X_pred = X_pred.astype(float)
	X_pred = ndimage.median_filter(X_pred, size=10)
	X_pred = ndimage.gaussian_filter(X_pred, sigma=5)

	# predict pH
	X_pred = X_pred.reshape(S)#*(X_pred_prob.reshape(S) > 0.20).astype(float)
	X3 = X[:,3].reshape(S)
	C = cm.rainbow((X_pred - 4.8) / (6.8-4.8)) 
	for i in range(3):
		C[:,:,i]*=(X3 > 1500.).astype(float)#*(X_pred_prob.reshape(S) > 0.25).astype(float)
		C[:,:,i]*=( X3 / 12000.*(X3 / 12000.<1).astype(float) + (X3 / 12000.>=1).astype(float))

	
	
	#########################################################################################
	# Save data
	#########################################################################################
	
	# Save intensity corrected data
	signal = (IM[2].reshape(S)*255./10000.).astype(np.uint8)
	cv2.imwrite(folder_out + "Intensity\\"+position+"\\_correct_Signal"+str(k)+".png", signal )
	# Save pH data
	pred = (X_pred*10).astype(np.uint8)
	cv2.imwrite(folder_out + "pH\\"+position+"\\_pred_pH"+str(k)+".png",pred)

	return 1