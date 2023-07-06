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
from PIL import Image
import cv2
#import LoadImagFiles

n_slice = 160

def run(params, absFilePath):
	print ("----------------------------------")
	print ("Apply colormap on the data ")
	print ("     ")
	print ("-----------------------------------")
	print ("")

	# sequence = params["sequence"]
	# channels = params["channels"]					
	# solv = params["solv"]			
	# data_sub = params["data_sub"]	
	# W = params["W"]					#100 						# number of hidden layers

	# # Base files
	# base = params["base"]			#"E:\\PROGRAMS\\SENSOILs\\calibration\\"
	# date_model = params["date calibration"]			#"20190626"  				#"20190822"
	# date = params["date data"]	
	# data_name = params["data_name"]
base_name = "0_pH_SOIL_"

####################################################################
# find direct directories
base = "Y:\\3 - PROCESS\\code\\pH_calibration\\Stitching\\output\\"
base_int = ""
base_pH = ""
dir_list = os.listdir(base)
for dir in dir_list:
	if "Int" in dir:
		base_int = base + dir + "\\"
	if "pH" in dir:
		base_pH = base + dir + "\\"

######################################################################	
for i in range(1,n_slice):	
	print i
	img_pH = (mpimg.imread(base_pH + base_name +str(i)+".tif"))[:,:,0]
	img_pH = ndimage.median_filter(img_pH, size=5)
	img_pH = ndimage.gaussian_filter(img_pH, sigma=15)	
	
	img_int = (mpimg.imread(base_int + base_name +str(i)+".tif"))[:,:,0]
	img_int = ndimage.median_filter(img_int, size=5)
	#img_int = ndimage.gaussian_filter(img_int, sigma=5)	
	
	C = cm.rainbow((img_pH - 48.) / (68.-48.)) 
	for k in range(3):
		min_threshold = 30.
		# remove background
		C[:,:,k]*=img_int * (img_int > min_threshold)#*(X_pred_prob.reshape(S) > 0.25).astype(float)
		# MOdulate intensity
		max_threshold = 250.
		C[:,:,k]*=( img_int / max_threshold*( img_int / max_threshold <1).astype(float) + ( img_int / max_threshold >= 1).astype(float))



	#########################################################################################
	# Save data
	#########################################################################################
	#im_rgb = cv2.cvtColor(C[:,:,[0,1,2]].astype(np.uint8), cv2.COLOR_RGB2BGR)
	im_bgr = cv2.merge([C[:,:,2],C[:,:,1],C[:,:,0]]).astype(np.uint8)
	#print  np.max(np.max(C[:,:,0] - C[:,:,1])),           " // ", np.min(np.min(C[:,:,0] - C[:,:,1]))
	#print  np.max(np.max(im_bgr[:,:,0] - im_bgr[:,:,1])), " // ", np.min(np.min(im_bgr[:,:,0] - im_bgr[:,:,1]))
	if i<10:
		#ret = mpimg.imsave(base + "00" + str(i)+".png", C[:,:,[0,1,2]])
		ret = cv2.imwrite(base + "00" + str(i)+".png", im_bgr)#.astype(np.uint8))
	elif i<100:
		#ret = mpimg.imsave(base + "0"  + str(i)+".png", C[:,:,[0,1,2]])
		ret = cv2.imwrite(base + "0" + str(i)+".png", im_bgr)#.astype(np.uint8))
	else:
		#ret = mpimg.imsave(base +        str(i)+".png", C[:,:,[0,1,2]])
		ret = cv2.imwrite(base + "" + str(i)+".png", im_bgr)#.astype(np.uint8))
	#print "COUCOU: ", np.shape(im_bgr)
	#print  np.max(np.max(im_bgr[:,:,0] - im_bgr[:,:,0]))
#X_pred = (X_pred*10.).astype(np.uint8)
#img = Image.fromarray(X_pred)
#img.save(base + "\\pred_pH"+str(k)+".png")
#mpimg.imsave(base + "\\pred_pH"+str(k)+".png", X_pred,cmap = 'gray')
#plt.imshow(C)
#plt.imshow(X_pred_masked,cmap = 'rainbow')
#plt.clim(4.8, 6.8)	
#plt.colorbar()

#plt.savefig(base + "\\vis"+str(k)+".png", bbox_inches="tight", dpi = 300)
#plt.close() 
#k+=1
