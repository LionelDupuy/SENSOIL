HOW TO CALIBRATE THE DATA
this file explains how to use the different tools to calibrate a model to predict the pH

A - Characterise how histogram varies as a function of z in the image (correct stich problems)
	A1 - Change the base directory in first line (where to find the image data)
	     Use the image data, not calibration data for that
	   - Run to open images

	A2 - Change output directory in first line
	   - to extract the histograms
	   - copy the .csv files in a folder with the name of the 
	   - redo A1 and A2 for a few Z values where there is only soil (to have more reps)

B - Get point of interest on the image (model not calibrated on all pixels, only selected points) 
	B1 - Change the base directory (dir_base) in first line (where to find the calibration data)
	   - Run to open images

	B2 - Run (Threshold image)

	B1 - Run to open image again

	B3 - Run (select a series of points)

MAIN.py - Fit the model on claibration data and process all images
	There are three steps I, II and III. Run them separately by commenting the other steps 
	If one step is successfull, no need to re-run it, move to the next step