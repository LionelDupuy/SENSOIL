run("Median...", "radius=4 stack");
setAutoThreshold("Otsu dark");
run("Threshold...");
run("Analyze Particles...", "clear add stack");
