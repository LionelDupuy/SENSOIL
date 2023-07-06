##################################################################################
# Load the image and apply correction for Z variations 
##################################################################################
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from scipy import ndimage
import numpy as np
import pickle
import sys


is_three = False
if (sys.version_info > (3, 0)):
	is_three = True
	
def Load(file,folder_correct = "", channel = "a"):
	if folder_correct == "":
		img=mpimg.imread(file)
	else: 
		mult = np.array(pickle.load(open(folder_correct+"CORRECT_p3.pkl","rb"))).astype(np.uint16)
		#mult = np.array(pickle.load(folder_correct+"CORRECT_p3.pkl")).astype(np.uint16)
		#CORRECT = 1700. / mult
		img=mpimg.imread(file)*1700./mult
	#img = ndimage.median_filter(img, size=10)
	#img = ndimage.gaussian_filter(img, sigma=sig)
	return img