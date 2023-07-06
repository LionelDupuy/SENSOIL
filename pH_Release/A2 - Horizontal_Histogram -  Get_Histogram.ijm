//dir_output = "E:/PROGRAMS/SENSOILs/calibration/histograms/"
 Dialog.create("Dialog");
 Dialog.addMessage("you must create a folder IN THE CODE\\HISTOGRAM FOLDER with THE DATE a name");
 Dialog.show()
 
dir_output = getDirectory("Choose a Directory for saving histograms"); 
im = getTitle();
nBins = 65536;


for (k=0;k<64;k++)
{
	selectWindow(im);
	makeRectangle(0, k*8, 512, 8);
	run("Histogram", "stack");
	selectWindow(im);
	getHistogram(value, count, nBins);
	
	run("Clear Results");

	for (i=0; i<nBins; i++){
		  //setResult("Value", i, value[i]);
		  setResult("Count", i, count[i]);
	}
	updateResults();
	saveAs("Results",dir_output + "histogram_"+k+".csv");

}
 run("Close All")
