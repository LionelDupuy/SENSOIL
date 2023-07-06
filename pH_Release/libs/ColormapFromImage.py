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


def run(params, absFilePath):
	print ("----------------------------------")
	print ("Apply colormap on the data ")
	print ("     ")
	print ("-----------------------------------")
	print ("")

	sequence = params["sequence"]
	channels = params["channels"]					
	solv = params["solv"]			
	data_sub = params["data_sub"]	
	W = params["W"]					#100 						# number of hidden layers

	# Base files
	base = params["base"]					# where the data is
	base_output  = params["base_output"]	# where process outputs are	
	date_model = params["date calibration"]			#"20190626"  				#"20190822"
	date = params["date data"]	
	data_name = params["data_name"]
	
	
	# thresholds
	min_threshold = params["min_threshold"]#30.
	max_threshold = params["max_threshold"]#250.
	n_slice =  params["n_slice"] #120

	####################################################################
	
	folder = base_output + date + "\\" + data_name + "\\Stitch\\"
	dirs = os.listdir(folder)
	for dir1 in dirs:	
		root1 = folder + dir1 + "\\"
		dir_z = os.listdir(root1)
		for dir2 in dir_z:
			# find direct directories
			base_name = "0_pH_SOIL_"
			base_int = ""
			base_pH = ""
			folder_out = folder + dir1 + "\\" + dir2 + "\\" 
			dirs2 = os.listdir(folder + dir1 + "\\" + dir2 + "\\")
			for dir in dirs2:
				if "Int" in dir:
					base_int = folder + dir1 + "\\" + dir2 + "\\" + dir + "\\"
				if "pH" in dir:
					base_pH = folder + dir1 + "\\" + dir2 + "\\" + dir + "\\"

			######################################################################	
			for i in range(1,n_slice):	
				print i
				img_pH = (mpimg.imread(base_pH + base_name +str(i)+".tif"))[:,:,0]
				img_pH = ndimage.median_filter(img_pH, size=5)
				img_pH = ndimage.gaussian_filter(img_pH, sigma=15)	
				
				img_int = (mpimg.imread(base_int + base_name +str(i)+".tif"))[:,:,0]
				img_int = ndimage.median_filter(img_int, size=5)

				
				C = cm.rainbow((img_pH - 48.) / (68.-48.)) 
				for k in range(3):
					# remove background
					C[:,:,k]*=img_int * (img_int > min_threshold)#*(X_pred_prob.reshape(S) > 0.25).astype(float)
					# MOdulate intensity
					C[:,:,k]*=( img_int / max_threshold*( img_int / max_threshold <1).astype(float) + ( img_int / max_threshold >= 1).astype(float))



				#########################################################################################
				# Save data
				#########################################################################################
				im_bgr = cv2.merge([C[:,:,2],C[:,:,1],C[:,:,0]]).astype(np.uint8)
				if i<10:
					ret = cv2.imwrite(folder_out + "00" + str(i)+".png", im_bgr)
				elif i<100:
					ret = cv2.imwrite(folder_out + "0" + str(i)+".png", im_bgr)
				else:
					ret = cv2.imwrite(folder_out + "" + str(i)+".png", im_bgr)
