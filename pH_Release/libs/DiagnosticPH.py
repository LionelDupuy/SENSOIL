import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
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
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import axes3d
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold

import LoadImagFiles
########################################################################
# Ideas
# Use unsupervised classification to select only reliable pixels?
#   1 - Cross validation
#	2 - Use score to plot best model then combined model	
#	2 - same number of points per pH value
#	3 - why effect of "depth" not working (clear gradiant in the image)
#	3 - Why dimentionality does not overfit the data
#   4 - Use probabilities to select best model)
# GET MORE AND BETTER DATA
# Use classification instead of regression to predict pH
# 
###########################################################################
sub_smpl = 20
threshold = 0
s_im_smpl = 0
base_index = 3
is_raw_data = True
is_trained_data = True

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure(1)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples",fontsize = fontsize)
    plt.ylabel("Score",fontsize = fontsize)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.xticks(size = fontsize)
    plt.yticks(size = fontsize)
    plt.legend(loc="best")
    return plt

	
############################################################################
# Get the data files
###########################################################################
base = "E:\\PROGRAMS\\SENSOILs\\calibration\\"
folder = base + "20190626_Daniel_pHChange\\"
folder_correct = base + "histograms\\"
dir_tot = os.listdir(folder)
del dir_tot[-1]
dirs = dir_tot
print "----------------------------------"
print "BASE FOLDER USED FOR ANALYSIS: "
print "     ", folder
print "-----------------------------------"
print ""
dir0 = ""


channels = ["a","b","c","d"]
DATA = []
DATA_TOT = np.array([])

PH = np.array([])
list_pH = []
#--------------------------------------------------------------------------------------
# Get image size
print "get image size ..."
file = folder + dirs[0] + "\\LOOP_0\\Z_Window_z1\\"+channels[0]+"_0.tif"
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
	
for j in range(3):
	#--------------------------------------------------------------------------------------
	# Read maxima points
	PTS_X = [() for i in range(len(dirs))]
	PTS_Y = [() for i in range(len(dirs))]

	f = open("Point_Selection_on_coating" +str(j+1) + ".csv")
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
	print "reading data ..."
	for i in range(len(dirs)):
		d = dirs[i]
		block_c = []
		num = (d.split("pH")[2]).split("_")
		pH = float(num[0]) + 0.01*float(num[1]) 
		
		#index = np.random.permutation(127) 
		#print PTS_X[i][(1,2,3,4,5,6,7,8,9)]
		ptsX = np.array(PTS_X[i])#[index]
		ptsY = np.array(PTS_Y[i])#[index]
		print "     pH processed: ", pH
		
		
		D = []
		D.append(np.array(ptsX))
		D.append(np.array(ptsY))
		for c in channels:
			file = folder + d + "\\LOOP_0\\Z_Window_z" + str(j + 1) + "\\"+c+"_0.tif"
			img=LoadImagFiles.Load(file, folder_correct, c)#mpimg.imread(file)
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

# ############################################################################
# # Machine learning
# ###########################################################################
n_input = 4
fontsize = 18
data_sub = 5
sequence = [1,2,3,4,5]
X = DATA_TOT[sequence,::data_sub].transpose().astype('float')
Y = DATA_TOT[6,::data_sub].astype('float')

if is_trained_data == True:
	print "Regression"	
	# Pre-process (PCA + normalisation)
	pca = decomposition.PCA(n_components=n_input)
	Xp = X
	pca.fit(Xp)
	Xp = pca.transform(Xp)
	pickle.dump(pca, open( "pca.p", "wb" ) )

		
	y = (Y - Y.min()) / (Y.max() - Y.min())
	sc = StandardScaler()
	sc.fit(Xp)
	Xp = sc.transform(Xp)	
	pickle.dump(pca, open( "pca.p", "wb" ) )
	
	yp = Y.astype('float')
	SCOR = []
	
	#--------------------------------------------------------------------------------------
	# PLot learning curve
	# It shows if adding data is going to benefit AND if there is a bias / variance pb
	# If converge to the same value - adding data does not help but you may reduce the bias by complexifying the dataset
	# If they don't converge to the same value, adding data improve generalisation
	
	# clf = MLPRegressor(alpha=0.2, max_iter=2000, hidden_layer_sizes = (150,150,150))
	# title = "NN learning curve"
	# plot_learning_curve(clf, title, Xp, yp, ylim=None, cv=ss,
    #                     n_jobs=4, train_sizes=np.linspace(.1, 1.0, 4))
	
	
	
	#--------------------------------------------------------------------------------------
	# Plot validation curve
	# helps determine when the model starts to overfit
	n_splits = 6
	#ss = ShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=0)
	ss = KFold(n_splits=n_splits)
	n = 5
	
	mult = (np.logspace(0, 1.5, num=n)+0.5).astype('int')
	#mult = [350,450]
	pickle.dump(mult, open('NUM_LAYERS.p', 'wb'))

	if True:
		TRAIN = [[] for i in mult]
		TEST = [[] for i in mult]
		TRAIN_STD = []
		TEST_STD = []
		for i in range(n):
			layer = (mult[i], mult[i], mult[i])
			print "Layer: ", layer
			clf = MLPRegressor(alpha=0., max_iter=2000, hidden_layer_sizes = layer, learning_rate_init = 0.001, solver = 'lbfgs',learning_rate = 'adaptive')
			score_test = []
			score_train = []
			
			for DAT in ss.split(Xp):
				DAT[0]
				X_train = Xp[DAT[0],:]
				Y_train = yp[DAT[0]].astype('float')	
				X_test = Xp[DAT[1],:]
				Y_test = yp[DAT[1]].astype('float')	

				# Learn model
				clf.fit(X_train, Y_train)
				#score_test.append(clf.score(X_test, Y_test))
				#score_train.append(clf.score(X_train, Y_train))
				TRAIN[i].append(clf.score(X_train, Y_train)) # np.sum(np.power((clf.predict(X_train) - Y_train),2)))#
				TEST[i].append(clf.score(X_test, Y_test)) # np.sum(np.power((clf.predict(X_test) - Y_test),2)))#

		pickle.dump(TRAIN, open('VALIDCURV_TRAIN.p', 'wb'))
		pickle.dump(TEST, open('VALIDCURV_TEST.p', 'wb'))
	TRAIN0 = np.array(pickle.load(open('VALIDCURV_TRAIN.p', 'rb')))
	TEST0 = np.array(pickle.load(open('VALIDCURV_TEST.p', 'rb')))
	mult = np.array(pickle.load(open('NUM_LAYERS.p', 'rb')))
	
	TRAIN = np.mean(TRAIN0,axis = 1)
	TEST = np.mean(TEST0,axis = 1)
	TRAIN_STD = np.std(TRAIN0,axis = 1)
	TEST_STD = np.mean(TEST0,axis = 1)
	
	plt.figure(2)
	plt.title("Validation Curve")
	plt.xlabel("Number of elements in layer", fontsize = fontsize)
	plt.ylabel("Score", fontsize = fontsize)
	#plt.ylim(0.0, 1.1)
	lw = 2
	plt.semilogx((mult)[1:], (TRAIN )[1:], 'o-', color="r", lw=lw, label="Training score")
	plt.fill_between((mult)[1:], (TRAIN - TRAIN_STD)[1:],
					 (TRAIN + TRAIN_STD)[1:], alpha=0.2,
					 color="r", lw=lw)
	plt.semilogx((mult)[1:], (TEST)[1:], 'o-', color="g", lw=lw, label="Cross-validation score")
	plt.fill_between((mult)[1:], (TEST - TEST_STD)[1:],
					 (TEST + TEST_STD)[1:], alpha=0.2,
					 color="g", lw=lw)
	plt.legend(loc="best", fontsize = fontsize)	
	plt.xticks(size = fontsize)
	plt.yticks(size = fontsize)
	plt.grid()
	plt.show()	

	
