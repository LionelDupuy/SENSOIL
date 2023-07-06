# deal with correction / option
# Read me
import os
import sys

is_three = False
if (sys.version_info > (3, 0)):
	is_three = True
############################################################################
# Model parameters
############################################################################
params = {
"sequence" : [0,2,3,4,5],		# [2,3,4,5]
								# 0 - z ; 
								# 1 - x (do not use)
								#2 - 488 (a) ; 3 - 512 (b, broken); 4 - 564 (c); 5 - 635 (d)
								# change only x / y to check if it improves
"channels" : ["a","b","c","d"],	# names of the channels: do not change

"W" : 400, 						# number of hidden layers
"solv" : 'lbfgs', #'adam'#		# type of solver '
"data_sub" : 5,					# sub sampling of the data to save time

# base folders where to load and save data Y:\1 - DUMP\Daniel\pH_Data  Y:\\2 - RAW\\Daniel\\pH_Data\\
"base" : "D:\\PROGRAMMING\\SENSOIL\\calibration_ph\\",		# DANIEL: PUT ALL YOU PH DATA INTO ONE FOLDER WITH THE DATE IN IT as "20190626"
													#         it's better you don't change too much this for simplicity

#"base_output" :"Y:\\3 - PROCESS\\pH\\",				# this is where the data is being saved 
"base_output" :"D:\\PROGRAMMING\\SENSOIL\\5 - CODE\\pH_calibration\\Res",
# Name of data to process, date of calibration and data collected
"data_name" : "Daniel_pHTimelaps",
"date calibration" : "20190626", # this is from calibration, if it is good, keep it. This is from B (Fiji) ROI folder
"date data" : 		 "2020021023", #"20190626",  # do calibration and data acquisition on the same day if possible, this is from yhe plant data, From A (Fiji) histogram forlder

# colormaps, numbers of slices etc...
"min_threshold": 30.,
"max_threshold": 250.,
"n_slice":150
}
absFilePath = os.path.dirname(os.path.abspath(__file__)) + "\\"
############################################################################
# 0 - Ratiometrics
############################################################################
#from libs import Ratiometrics
#Ratiometrics.run(params, absFilePath)

############################################################################
# I - Correct intensity based on vertical histogram
############################################################################
#from histograms import Correct_Horizontal_Histogram
#Correct_Horizontal_Histogram.run(params, absFilePath)

############################################################################
# II - Learn pH model
############################################################################
# !!!!! currently only reads the first three calibration data - need to correct
from libs import LearnPH
LearnPH.run(params, absFilePath)

############################################################################
# III - Process the full scan
############################################################################
#from libs import ProcessPH
#ProcessPH.run(params, absFilePath)

############################################################################
# IV - Stich the data
############################################################################
# When doind the stiching, save the output in an output folder 
#		in the base folder
#		under the correct data
#		in a folder name stitch_output

############################################################################
# V - Apply colormap on the data
############################################################################
# #check min / max threshold for visualisation
# from libs import ColormapFromImage
# ColormapFromImage.run(params, absFilePath)