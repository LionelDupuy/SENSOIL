##################################################################################
# This code use a NN model fitted to predict pH and process bulks of images with it.
# The code write two files. 8bit image which is pH*10. An 8 bit color image which is a colormap of the pH 
##################################################################################

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from scipy import ndimage
import numpy as np
import os
from scipy import stats#.linregress
import pickle
import PredictPH
import sys


is_three = False
if (sys.version_info > (3, 0)):
	is_three = True
	
fontsize = 18

sig = 1


def run(params, absFilePath):
	print ("----------------------------------")
	print ("Process scan data: ")
	print ("     ")
	print ("-----------------------------------")
	print ("")
	###########################################################################
	is_calibration_data = False
	############################################################################
	# Get the parameters
	###########################################################################

	sequence = params["sequence"]
	channels = params["channels"]					
	solv = params["solv"]			
	data_sub = params["data_sub"]	
	W = params["W"]					#100 						# number of hidden layers

	# Base files
	base = params["base"]			#"E:\\PROGRAMS\\SENSOILs\\calibration\\"
	base_output  = params["base_output"]
	date_model = params["date calibration"]			#"20190626"  				#"20190822"
	date = params["date data"]	
	data_name = params["data_name"]

	# Check output folder is not already there
	folder_out = base_output + date + "\\" + data_name + "\\" 
	if not os.path.isdir(base_output + date + "\\"):
		os.mkdir(base_output + date + "\\")
	if not os.path.isdir(folder_out):
		os.mkdir(folder_out)
	if not os.path.isdir(folder_out + "Intensity\\"):
		os.mkdir(folder_out + "Intensity\\")
	if not os.path.isdir(folder_out + "pH\\"):
		os.mkdir(folder_out + "pH\\")
	if not os.path.isdir(folder_out + "pHColomap\\"):
		os.mkdir(folder_out + "pHColomap\\")		
	if not os.path.isdir(folder_out + "Stitch\\"):
		os.mkdir(folder_out + "Stitch\\")	
	############################################################################
	# Get relevant files and folders
	###########################################################################

	folder = base + date + "_" +  data_name + "\\"
	dir_tot = os.listdir(folder)
	if len(dir_tot) > 10:
		del dir_tot[-1]
		
	folder_correct = absFilePath + "histograms\\"
	model_code = solv
	xy = []
	if 0 in sequence: xy.append("X")
	if 1 in sequence: xy.append("Y")
	model_code += str(len(sequence) - len(xy)) + "F_"
	for s in xy: model_code += s
	model_code += "_W" + str(W)

	dir_models = absFilePath + "models\\" + date_model + "\\" + model_code + "\\"	
	if True:
		print ("----------------------------------")
		print ("PROCESS IMAGE DATA FILES: ")
		print ("     ", folder)
		print ("-----------------------------------")
		print ("")

	dir0 = ""

	# ############################################################################
	# # Load Model
	# ###########################################################################
	clf = pickle.load(open(dir_models + "Neural Net"))
	pca = pickle.load(open(dir_models + "pca.p"))
	sc  = pickle.load(open(dir_models + "transform.p"))	

	# ############################################################################
	# Get channels
	# ############################################################################
	ch = []
	for c in channels:
		if os.path.isfile(folder + dir_tot[0] + "\\Z_Window_z1\\"+c+"_0.tif"):
			ch.append(c)
	print ("Channels detected: " , ch)
	# ############################################################################
	# Get image size
	# ############################################################################

	print ("get image size ...")
	file = folder + dir_tot[0] + "\\Z_Window_z1\\"+ch[0]+"_0.tif"
	img=mpimg.imread(file)
	S = np.shape(img)
	print ("Image size= ", S)

	# ############################################################################
	# # Predict pH 
	# ###########################################################################
		
	for dir1 in dir_tot:
		# Create subfolder for depth
		if not os.path.isdir(folder_out + "pHColomap\\"+dir1+"\\"):
			os.mkdir(folder_out + "pHColomap\\"+dir1+"\\")
			print (dir1)
		if not os.path.isdir(folder_out + "pH\\"+dir1+"\\"):
			os.mkdir(folder_out + "pH\\"+dir1+"\\")
		if not os.path.isdir(folder_out + "Intensity\\"+dir1+"\\"):
			os.mkdir(folder_out + "Intensity\\"+dir1+"\\")
		
		
		if is_calibration_data:
			root1 = folder + dir1 + "\\LOOP_0\\"
		else:
			root1 = folder + dir1 + "\\"
		dir_z = os.listdir(root1)

				
		for dir2 in dir_z:
			# Create subfolder for depth
			if not os.path.isdir(folder_out + "pHColomap\\"+dir1+"\\"+dir2+"\\"):
				os.mkdir(folder_out + "pHColomap\\"+dir1+"\\"+dir2+"\\")
				
			if not os.path.isdir(folder_out + "pH\\"+dir1+"\\"+dir2+"\\"):
				os.mkdir(folder_out + "pH\\"+dir1+"\\"+dir2+"\\")
			if not os.path.isdir(folder_out + "Intensity\\"+dir1+"\\"+dir2+"\\"):
				os.mkdir(folder_out + "Intensity\\"+dir1+"\\"+dir2+"\\")
			
			print ("Directory:", dir2)
			folder_in = root1 + dir2 + "\\"
			k = 0
			file = folder_in+"a_"+str(k)+".tif"
			while os.path.isfile(file):
				print ("processing:  ", file)
				ret = 0
				count = 0
				while ret==0 and count<10:
					ret = PredictPH.Predict(clf,pca, sc,folder_in, folder_out, dir1+"\\"+dir2, folder_correct , k, sequence ,ch)
					count += 1
				k+=1
				file = folder_in+"a_"+str(k)+".tif"