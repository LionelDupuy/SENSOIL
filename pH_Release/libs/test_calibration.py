import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from scipy import ndimage
import numpy as np
import os
from scipy import stats#.linregress
import pickle
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import axes3d
import LoadImagFiles

base = "E:\\PROGRAMS\\SENSOILs\\calibration\\"
date = "20190822"
folder = base + date +"_Daniel_pHContr\\" #"20190626_Daniel_pHChange\\"  # 
date = "20190626" #"20190822" #  
folder_models = base + "models\\" + date + "\\lbfgs4F_X_W100\\"	#"models\\lbfgs4F_XY_W200\\"
folder_correct = base + "histograms\\"
dir_tot = os.listdir(folder)
del dir_tot[-1]
dirs = dir_tot

sig = 1
fontsize = 18

PH = np.array([])
list_pH = []

def plot_maps_pH(dirs,clf,pca,sc, sequence):
	sub = 3
	indices = np.arange(0,len(dirs),sub)

	#fig, ax = plt.subplots(1, len(indices))
	f = plt.figure()
	gs = gridspec.GridSpec(3, len(indices), height_ratios=[10,10,1], hspace=0.15)
	#gs1.update(left=0.05, right=0.48, wspace=0.05)
	for i in indices:
		d = dirs[i]
		print d
		# load the model
		IM = []
			
		
		# Add x and y coordinates as input variables
		xx, yy = np.meshgrid(np.arange(S[0]),
								 np.arange(S[1]))
		IM.append(xx.ravel())
		IM.append(yy.ravel())
			
		# Add wavelength as input variables
		for c in channels:
			file = folder + d + "\\LOOP_0\\Z_Window_z3\\"+c+"_0.tif"
			img = LoadImagFiles.Load(file, folder_correct)#mpimg.imread(file)
			#img = ndimage.median_filter(img, size=10)
			img = ndimage.gaussian_filter(img, sigma=sig)
			IM.append(img.ravel())

		X = np.array(IM).transpose()

		# transform the data
		Xp2 = X[:,sequence]
		if pca != None:
			Xp2  = pca.transform(Xp2)						# PCA
		if sc!= None:
			Xp2 = sc.transform(Xp2)						# Normalise
		
		# Prediction by block to avoid memory errors
		X_pred = X[:,0]*0.
		X_pred_prob = X[:,0]*0.
		nn = len(X_pred)
		SUB = 10
		
		#X_pred = clf.predict(Xp2).astype(float)		
		for kk in range(SUB):
			X_pred[kk*nn/SUB:(kk+1)*nn/SUB] = clf.predict(Xp2[kk*nn/SUB:(kk+1)*nn/SUB,:]).astype(float) # [kk*nn/SUB:(kk+1)*nn/SUB,:]


		X_pred = ndimage.gaussian_filter(X_pred.reshape(S), sigma=5).ravel()
		
		
		RGB = plt.cm.rainbow((np.clip(X_pred.reshape(S),5,7)-5.)/2.)
		X3 = X[:,3].reshape(S)
		dI = 0.5
		di = 0.35
		RGB[:,:,3] = ( np.clip(X3 / float(np.max(np.max(X3))),di,1.-dI) - di ) / (1. - dI - di) 

		ii = i/sub
		# Plot predicted pH
		ax = plt.Subplot(f, gs[1, ii])
		ax.set_title("pH " + str(list_pH[i]), fontsize=fontsize)
		if i<len(dirs)-1:
			im = ax.imshow(RGB)#,cmap = 'rainbow')
			#im.set_clim(5, 7)	
			#plt.colorbar(im, ax=ax[ii])
		else :#i==len(dirs)-1:
			im = ax.imshow(RGB)#,cmap = 'rainbow')
			#im.set_clim(5, 7)
			#plt.colorbar(im, ax=ax[ii])
		#make_ticklabels_invisible(plt.gcf())
		ax.axis('off')
		f.add_subplot(ax)
		
	# Plot wavelength
	ii = 0
	cmaps = ['Blues', 'Greens', 'Oranges', 'Reds']
	lambas = [488,512,562,632]
	for ii in range(len(channels)):
		#file = folder + dirs[0] + "\\LOOP_0\\Z_Window_z2\\"+c+"_0.tif"
		#img = LoadImagFiles.Load(file, folder_correct)#mpimg.imread(file)
		#img = ndimage.gaussian_filter(img, sigma=sig)
		#X = np.array(img)
		XX = X[:,2+ii]*5.
		ax = plt.Subplot(f, gs[0, ii])
		ax.set_title(str(lambas[ii]) + ' nm', fontsize=fontsize)
		im = ax.imshow(XX.reshape(S),cmap = 'bone')
		im.set_clim(2000.,20000.)
		ax.axis('off')
		f.add_subplot(ax)
		#make_ticklabels_invisible(plt.gcf())
		ii+=1
		
	#make_ticklabels_invisible(plt.gcf())
	ax = plt.Subplot(f, gs[2, :])
	cmap = mpl.cm.rainbow
	norm = mpl.colors.Normalize(vmin=5, vmax=7)
	cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
									norm=norm,
									orientation='horizontal')
	cb1.set_label('pH', fontsize=fontsize)
	for ii in range(2):
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(fontsize)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(fontsize)
	f.add_subplot(ax)
	plt.show()

#--------------------------------------------------------------------------------------
# Get image size
channels = ["a","b","c","d"]
sequence = [1,2,3,4,5]
print "get image size ..."
file = folder + dirs[0] + "\\LOOP_0\\Z_Window_z2\\"+channels[0]+"_0.tif"
img=LoadImagFiles.Load(file, folder_correct)#mpimg.imread(file)
S = np.shape(img)
print "Image size= ", S
#--------------------------------------------------------------------------------------
# Get pH values
for d in dirs:
	block_c = []
	num = (d.split("pH")[2]).split("_")
	pH = float(num[0]) + 0.01*float(num[1]) 
	print "     pH found: ", pH
	list_pH.append(pH)	

#--------------------------------------------------------------------------------------
# Get model
clf = pickle.load(open(folder_models + "Neural Net"))
pca = pickle.load(open(folder_models + "pca.p"))
sc  = pickle.load(open(folder_models + "transform.p"))	
#--------------------------------------------------------------------------------------

plot_maps_pH(dirs,clf,pca,sc,sequence)