 Dialog.create("Dialog");
 Dialog.addMessage("you must create a folder IN THE CODE\\ROI FOLDER with 'DATE' as name. Save results as Point_Selection_on_coating1.csv");
 Dialog.show()
 
//dir_base = "Y:\\1 - DUMP\\Daniel\\pH_Data\\20190722_Daniel_PlantpH_Calibration\\4_97";
//dir_base = "Y:\\1 - DUMP\\Daniel\\pH_Data\\20190626_Daniel_pHChange";

dir_base=getDirectory("Choose a Directory"); 

dir_add = "\\LOOP_0\\Z_Window_z1\\a_0.tif";

pH = newArray("4_97", "5_20" , "5_37","5_58", "5_78","6_03","6_18","6_38","6_63","6_80","6_97");

//setBatchMode(true);
for (i=0; i<pH.length; i++) {
    //showProgress(i+1, pH.length);
    print(dir_base + pH[i] + dir_add);
    open(dir_base + pH[i] + dir_add);
    //log(""+pH[i]);
}
run("Images to Stack");

/*run("Enhance Contrast...", "saturated=0 normalize equalize process_all");
setAutoThreshold("Yen");
setOption("BlackBackground", true);
run("Convert to Mask", "method=Yen background=Light calculate");
run("Analyze Particles...", "add stack");
*/