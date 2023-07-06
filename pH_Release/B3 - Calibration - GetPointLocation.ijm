run("8-bit");
run("Set Measurements...", "stack redirect=None decimal=3");
count=roiManager("count");
for (i=0; i<count; i++) { 
    roiManager("Select", i);
    run("Add Noise", "slice");
	run("Gaussian Blur...", "sigma=0.8 slice");
	run("Find Maxima...", "noise=0 output=[Point Selection]");
	roiManager("Add");
}
for (i=count-1; i>-1; i--) { 
    roiManager("Select", i);
    roiManager("Delete");
}