dir_base=getDirectory("Choose a Directory"); 
//Y:\1 - DUMP\Daniel\20191202555_Danbiel_pHTimeLaps
//dir_base = "Y:\\1 - DUMP\\Daniel\\2019Aug\\20190822_Daniel_PlantpH1\\LOOP_0\\";
//dir_base = "\\143.234.97.182\\SensoilData\\1 - DUMP\\Daniel\\2019Aug\\20190822_Daniel_PlantpH1\\LOOP_0\\"
dir_add = "Z_Window_z";


//setBatchMode(true);
for (i=1; i<11; i++) {
    dir = dir_base + dir_add + i + "\\";
    for (j=0; j<160; j++) {
    	print("    " + dir + "a_" + j + ".tif");
    	open(dir + "a_" + j + ".tif");
    }
}
run("Images to Stack");

/*run("Enhance Contrast...", "saturated=0 normalize equalize process_all");
setAutoThreshold("Yen");
setOption("BlackBackground", true);
run("Convert to Mask", "method=Yen background=Light calculate");
run("Analyze Particles...", "add stack");
*/