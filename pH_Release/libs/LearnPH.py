##################################################################################
# This code fit the ANN model
# Provide various plots of the model 
##################################################################################
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
from . import LoadImagFiles
import sys


is_three = False
if (sys.version_info > (3, 0)):
	is_three = True
	
	
fontsize = 18
is_raw_data = True
is_trained_data = True
sig = 1


X = []

def run(params, absFilePath):
	global sequence, channels, W, solv, data_sub, base, date
	global dirs, folder, folder_correct
	global S, list_pH

	print ("----------------------------------")
	print ("TRAIN NEURAL NET FOR PH PREDICTION: ")
	print ("     ")
	print ("-----------------------------------")
	print ("")

	############################################################################
	# Model parameters
	############################################################################
	sequence = params["sequence"] 
									# change only x / y to check if it improves
	channels = params["channels"]	#["a","b","c","d"]	# names of the channels: do not changes
	
	W = params["W"]					#100 						# number of hidden layers
	solv = params["solv"]			#'lbfgs' #'adam'#			# type of solver 'sgd', 
	data_sub = params["data_sub"]	#5					# sub sampling of the data to save time

	# Base files
	base = params["base"]			#"E:\\PROGRAMS\\SENSOILs\\calibration\\"
	date = params["date calibration"]			#"20190626"  				#"20190822"
	
	############################################################################
	# Get info from files
	###########################################################################

	folder = base + date+"_Daniel_pHContr\\"
	
	dirs = os.listdir(folder)
	dirs.sort()
	del dirs[-1]								# remove the last pH (because it's weird)

	# Model
	model_code = solv
	xy = []
	if 0 in sequence: xy.append("X")
	if 1 in sequence: xy.append("Y")
	model_code += str(len(sequence) - len(xy)) + "F_"
	for s in xy: model_code += s
	model_code += "_W" + str(W)

	folder_correct = absFilePath + "histograms\\"
	dir_models = absFilePath + "models\\" + date + "\\" + model_code + "\\"
	print( dir_models)
	if not os.path.isdir(dir_models):
		os.mkdir(dir_models)

	# ROI
	dir_ROI = absFilePath + "ROI\\" + date + "\\"



	#############################################################################
	# Initiate data structure
	#############################################################################
	n_input =  len(sequence)
	dir0 = ""
	DATA = []
	DATA_TOT = np.array([])
	PH = np.array([])
	list_pH = []

	#--------------------------------------------------------------------------------------
	# Get image size
	file = folder + dirs[0] + "\\LOOP_0\\Z_Window_z1\\"+channels[0]+"_0.tif"
	img=LoadImagFiles.Load(file, folder_correct)#mpimg.imread(file)
	S = np.shape(img)

	#--------------------------------------------------------------------------------------
	# Get pH values
	
	for d in dirs:
		block_c = []
		num = (d.split("pH")[2]).split("_")
		pH = float(num[0]) + 0.01*float(num[1]) 
		print ("     pH found: ", pH)
		
		list_pH.append(pH)	

	############################################################################
	# Read data
	###########################################################################
	csv_files = os.listdir(dir_ROI)
	for j in range(len(csv_files)):
		print ("Get Calibration data ", j)
		
		#--------------------------------------------------------------------------------------
		# Read maxima points
		PTS_X = [() for i in range(len(dirs))]
		PTS_Y = [() for i in range(len(dirs))]

		f = open(dir_ROI + "Point_Selection_on_coating" +str(j+1) + ".csv")
		for line in f:
			row = line.split(",")
			if "Slice" in row[3]:
				pass
			elif int(row[3]) == 0 or int(row[3]) == 11 :
				pass
			else:
				ind = int(row[3]) - 1
				PTS_X[ind] = PTS_X[ind] + (int(row[1]),)
				PTS_Y[ind] = PTS_Y[ind] + (int(row[2]),)

		#--------------------------------------------------------------------------------------
		for i in range(len(dirs)):
			d = dirs[i]
			block_c = []
			num = (d.split("pH")[2]).split("_")
			pH = float(num[0]) + 0.01*float(num[1]) 
			

			ptsX = np.array(PTS_X[i])#[index]
			ptsY = np.array(PTS_Y[i])#[index]
			
			
			D = []
			D.append(np.array(ptsX))
			D.append(np.array(ptsY))
			for c in channels:
				file = folder + d + "\\LOOP_0\\Z_Window_z" + str(j + 1) + "\\"+c+"_0.tif"
				img = LoadImagFiles.Load(file, folder_correct,c)
				#img=mpimg.imread(file)
				##img = ndimage.median_filter(img, size=10)
				#img = ndimage.gaussian_filter(img, sigma=sig)
				D.append(np.array(img[(ptsX,ptsY)]))
			D.append(np.array((ptsX))*0+pH)
			D = np.array(D)
			
			if pH < 7.2:
				if len(DATA_TOT) == 0:
					rng = [0]
					DATA_TOT = D
				else:
					rng = [len(DATA_TOT[0,:])]
					DATA_TOT = np.concatenate((DATA_TOT,D),axis = 1)
				rng.append(len(DATA_TOT[0,:]) - 1)
				DATA.append(rng)
			##############################################################################
			#np.save ("X", DATA)
			#np.save ("Y", PH)
			#np.save ("list_pH", list_pH)
			#np.save ("s_im_smpl", s_im_smpl)
			#DATA = np.array(DATA)
		f.close()
		



	############################################################################
	# Ratiometric Analysis
	###########################################################################
	# from sklearn.linear_model import LinearRegression	
	# Xr = DATA_TOT[2,::data_sub]/DATA_TOT[5,::data_sub]

	##model = SVR(C=0.01)## KernelRidge(alpha=1.0)# LinearRegression() #
	# model = MLPRegressor(alpha=1.0, max_iter=1000, hidden_layer_sizes = (9,9,9), activation = 'tanh')
	# model.fit(Xr.reshape(1,-1).transpose(), Y)
	# y_pred = model.predict(Xr.reshape(1,-1).transpose())

	# plt.figure(1)
	# ax = plt.subplot(1, 1, 1)
	# ax.plot(Xr, DATA_TOT[6,::data_sub] ,'+')
	# xx = np.arange(0.3,1.5,0.01).reshape(1,-1).transpose()
	# ax.plot(xx, model.predict(xx),'r-', linewidth = 3)
	# ax.set_xlabel("wavelength ratio",fontsize=fontsize)
	# ax.set_ylabel("pH",fontsize=fontsize)
	# for ii in range(2):
		# for tick in ax.xaxis.get_major_ticks():
			# tick.label1.set_fontsize(fontsize)
		# for tick in ax.yaxis.get_major_ticks():
			# tick.label1.set_fontsize(fontsize)

	# plt.figure(2)
	# ax = plt.subplot(1, 1, 1)
	# plot_actual_predict(ax, Y, y_pred, X)
	# for ii in range(2):
		# for tick in ax.xaxis.get_major_ticks():
			# tick.label1.set_fontsize(fontsize)
		# for tick in ax.yaxis.get_major_ticks():
			# tick.label1.set_fontsize(fontsize)

	# plt.show()
	# plot_maps_ratio(dirs,model, None, None,[2,5])
	# ############################################################################
	# # Learning
	# ###########################################################################
	# DATA_TOT indices


	X = DATA_TOT[sequence,::data_sub].transpose()
	Y = DATA_TOT[6,::data_sub]
    
    
	if is_trained_data == False:
		# Pre-process (PCA + normalisation)
		pca = decomposition.PCA(n_components=n_input)

		if n_input == 2:
			Xp = X
			pca.fit(Xp)
			pickle.dump(pca, open( dir_models + "pca.p", "wb" ) )
			Xp = pca.transform(Xp)
		else:
			Xp = X
			pca.fit(Xp)
			pickle.dump(pca, open( dir_models + "pca.p", "wb" ) )
			Xp = pca.transform(Xp)
		y = (Y - Y.min()) / (Y.max() - Y.min())
		sc = StandardScaler()
		sc.fit(Xp)
		pickle.dump(sc, open( dir_models + "transform.p", "wb" ) )

		X_train = sc.transform(Xp)	
		y_train = Y.astype('str')
		y_train = Y.astype('float')	
		X_test = X_train	
		y_test = Y.astype('str')		

		# Learn model
		names = [ "Neural Net", ]
		classifiers = [ MLPRegressor(alpha=0., max_iter=2000, hidden_layer_sizes = (W,W,W), learning_rate_init = 0.001, solver = solv,learning_rate = 'adaptive'),]

		x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
		y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
		h = .1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
								 np.arange(y_min, y_max, h))
		cm_RdBu = plt.cm.RdBu
		cm_bright = ListedColormap(['#FF0000', '#0000FF'])		
		i = 1				 
		for name, clf in zip(names, classifiers):
			#######################################################################################################
			# Fit and Save model
			clf.fit(X_train, y_train)
			#score = clf.score(X_test, y_test)
			
			pickle.dump(clf, open(dir_models + name,'wb'))
			
			# Plot the data is n == 2

			ax = plt.subplot(1, len(classifiers), i)
			y_data = Y.astype(float) #+ (np.random.rand(np.shape(Y)[0])  - 0.5) * 0.15
			y_pred = clf.predict(X_train).astype(float)
			plot_actual_predict(ax, y_data, y_pred, X)

		i += 1
		plt.show()




	# ############################################################################
	# # Test predictions
	# ###########################################################################
    # data vs prediction plot
	clf = pickle.load(open(dir_models + "Neural Net_p3.pkl", "rb"))
	pca = pickle.load(open(dir_models + "pca_p3.pkl", "rb"))
	sc  = pickle.load(open(dir_models + "transform_p3.pkl", "rb"))
    
	ax = plt.subplot(1, 1, 1)

	if n_input == 2:
		Xp = X
		pca.fit(Xp)
		pickle.dump(pca, open( dir_models + "pca.p", "wb" ) )
		Xp = pca.transform(Xp)
	else:
		Xp = X
		pca.fit(Xp)
		pickle.dump(pca, open( dir_models + "pca.p", "wb" ) )
		Xp = pca.transform(Xp)
	sc = StandardScaler()
	sc.fit(Xp)
	pickle.dump(sc, open( dir_models + "transform.p", "wb" ) )

	X_train = sc.transform(Xp)	
	y_train = Y.astype('str')
	y_train = Y.astype('float')	
	y_data = Y.astype(float) #+ (np.random.rand(np.shape(Y)[0])  - 0.5) * 0.15
	y_pred = clf.predict(X_train).astype(float)
	#export_predictions(y_data, y_pred)
	plot_actual_predict(ax, y_data, y_pred, X)

    # map
	plot_maps_pH(dirs, clf,pca,sc,sequence)

############################################################################
# Plot Functions
###########################################################################
def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)
def plot_actual_predict(ax, y_data, y_pred, X):
		z_pred = np.polyfit(y_data, y_pred,1)
		p = np.poly1d(z_pred)
		fit = p(y_data)
			
		# get the coordinates for the fit curve
		c_y = [np.min(fit),np.max(fit)]
		c_x = [np.min(y_data),np.max(y_data)]
			 
		# predict y values of origional data using the fit
		p_y = z_pred[0] * y_data + z_pred[1]
		 
		# calculate the y-error (residuals)
		y_err = y_pred -p_y
		 
		# create series of new test x-values to predict for
		p_x = np.arange(np.min(y_data),np.max(y_data)+1,1)
		 
		# now calculate confidence intervals for new test x-series
		mean_x = np.mean(y_data)         # mean of x
		n = len(y_data)              # number of samples in origional fit
		t = 2.31                # appropriate t value (where n=9, two tailed 95%)
		s_err = np.sum(np.power(y_err,2))   # sum of the squares of the residuals
		 
		confs = t * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((p_x-mean_x),2)/
					((np.sum(np.power(y_data,2)))-n*(np.power(mean_x,2))))))
		confs = t * np.sqrt( (s_err/(n-2)))
					
		# now predict y based on test x-values
		p_y = z_pred[0]*p_x+z_pred[1]
		 
		# get lower and upper confidence limits based on predicted y and confidence intervals
		lower = p_y - abs(confs)
		upper = p_y + abs(confs)
		 
		 
		# plot sample data
		color = ( X[:,1] - np.min(X[:,1]) ) / ( np.max(X[:,1]) - np.min(X[:,1]) )
		
		ax.scatter(y_data,y_pred,c=cm.rainbow(color), marker = '+', linewidth = 7, s = 10, edgecolors=None, alpha = 0.3)
		ax.set_xlabel("Data",fontsize=fontsize)
		ax.set_ylabel("Predicted",fontsize=fontsize)
		# plot line of best fit
		ax.plot(c_x,c_y,'r-', linewidth = 3)
		#ax.plot([5,7],[5,7],'g:', linewidth = 3, alpha = 0.7)
		#ax.axis('equal')
			
		ax.set_xlim((4.9, 7))
		ax.set_ylim((4.9, 7))
		ax.set_aspect('equal', 'box')		
		# plot confidence limits
		ax.plot(p_x,lower,'b--', linewidth = 3)
		ax.plot(p_x,upper,'b--', linewidth = 3)	
					
		for ii in range(2):
			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(fontsize)
			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(fontsize)
                
def export_predictions(Y_exp, Y_pred):
    DAT = np.transpose(np.array([Y_exp, Y_pred]))
    np.savetxt("Exp_Pred_ANN.txt", DAT, delimiter=',')#, fmt='%d'                
def plot_maps_pH(dirs, clf,pca,sc, sequence):
	
	sub = 3
	indices = np.arange(0,len(dirs),sub)

	#fig, ax = plt.subplots(1, len(indices))
	f = plt.figure()
	gs = gridspec.GridSpec(3, len(indices), height_ratios=[10,10,1], hspace=0.15)
	#gs1.update(left=0.05, right=0.48, wspace=0.05)
	for i in indices:
		d = dirs[i]
		# load the model
		IM = []
			
		
		# Add x and y coordinates as input variables
		xx, yy = np.meshgrid(np.arange(S[0]),
								 np.arange(S[1]))
		IM.append(xx.ravel())
		IM.append(yy.ravel())
			
		# Add wavelength as input variables
		for c in channels:
			file = folder + d + "\\LOOP_0\\Z_Window_z2\\"+c+"_0.tif"
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
			X_pred[int(kk*nn/SUB):int((kk+1)*nn/SUB)] = clf.predict(Xp2[int(kk*nn/SUB):int((kk+1)*nn/SUB),:]).astype(float) # [kk*nn/SUB:(kk+1)*nn/SUB,:]

		#X_pred = ndimage.median_filter(X_pred.reshape(S), size=10).ravel()
		X_pred = ndimage.gaussian_filter(X_pred.reshape(S), sigma=10).ravel()
		
		# predict pH
		#X_pred_masked =  X_pred * (X[:,3] > 3200.*(1. + float(i)/9.5))
		#X_pred_masked = np.ma.masked_where(X_pred_masked < 1. , X_pred_masked, copy = True)  # X[:,1] < 5000. and X_pred_prob > 0.3
		#X_pred_masked = X_pred_masked.reshape(S)
		
		RGB = plt.cm.rainbow((np.clip(X_pred.reshape(S),5,7)-5.)/2.)
		X3 = X[:,3].reshape(S)
		dI = 0.5
		di = 0.07
		RGB[:,:,3] = ( np.clip(X3 / float(np.max(np.max(X3))),di,1.-dI) - di ) / (1. - dI - di) 

		ii = int(i/sub)
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
	for c in channels:
		file = folder + dirs[0] + "\\LOOP_0\\Z_Window_z2\\"+c+"_0.tif"
		img = LoadImagFiles.Load(file, folder_correct)#mpimg.imread(file)
		img = ndimage.gaussian_filter(img, sigma=sig)
		X = np.array(img)
		ax = plt.Subplot(f, gs[0, ii])
		ax.set_title(str(lambas[ii]) + ' nm', fontsize=fontsize)
		im = ax.imshow(X.reshape(S),cmap = 'bone')
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

def plot_maps_ratio(dirs,clf,pca,sc, sequence):
	sub = 3
	indices = np.arange(0,len(dirs),sub)

	#fig, ax = plt.subplots(1, len(indices))
	f = plt.figure()
	gs = gridspec.GridSpec(3, len(indices), height_ratios=[10,10,1], hspace=0.15)
	#gs1.update(left=0.05, right=0.48, wspace=0.05)
	for i in indices:
		d = dirs[i]
		# load the model
		IM = []
			
		
		# Add x and y coordinates as input variables
		xx, yy = np.meshgrid(np.arange(S[0]),
								 np.arange(S[1]))
		IM.append(xx.ravel())
		IM.append(yy.ravel())
			
		# Add wavelength as input variables
		for c in channels:
			file = folder + d + "\\LOOP_0\\Z_Window_z2\\"+c+"_0.tif"
			img = LoadImagFiles.Load(file, folder_correct)#mpimg.imread(file)
			#img = ndimage.median_filter(img, size=10)
			img = ndimage.gaussian_filter(img, sigma=sig)
			IM.append(img.ravel())

		X = np.array(IM).transpose()

		# transform the data
		Xp2 = ( X[:,sequence[0]] / X[:,sequence[1]] ).reshape(1,-1).transpose()

		
		# Prediction (by block if memory errors
		X_pred = clf.predict(Xp2)	
		#SUB = 10		
		#nn = len(X_pred)
		#for kk in range(SUB):
		#	X_pred[kk*nn/SUB:(kk+1)*nn/SUB] = clf.predict(Xp2[kk*nn/SUB:(kk+1)*nn/SUB,:]).astype(float) # [kk*nn/SUB:(kk+1)*nn/SUB,:]

		#X_pred = ndimage.median_filter(X_pred.reshape(S), size=10).ravel()
		X_pred = ndimage.gaussian_filter(X_pred.reshape(S), sigma=10).ravel()
		
		# predict pH
		#X_pred_masked =  X_pred * (X[:,3] > 3200.*(1. + float(i)/9.5))
		#X_pred_masked = np.ma.masked_where(X_pred_masked < 1. , X_pred_masked, copy = True)  # X[:,1] < 5000. and X_pred_prob > 0.3
		#X_pred_masked = X_pred_masked.reshape(S)
		
		RGB = plt.cm.rainbow((np.clip(X_pred.reshape(S),5,7)-5.)/2.)
		X3 = X[:,3].reshape(S)
		dI = 0.5
		di = 0.07
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
	for c in channels:
		file = folder + dirs[0] + "\\LOOP_0\\Z_Window_z2\\"+c+"_0.tif"
		img = LoadImagFiles.Load(file, folder_correct)#mpimg.imread(file)
		img = ndimage.gaussian_filter(img, sigma=sig)
		X = np.array(img)
		ax = plt.Subplot(f, gs[0, ii])
		ax.set_title(str(lambas[ii]) + ' nm', fontsize=fontsize)
		im = ax.imshow(X.reshape(S),cmap = 'bone')
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

	