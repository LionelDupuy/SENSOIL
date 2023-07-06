run("Fit Spline");
getSelectionCoordinates(x, y);
l = 0.0;
print ('x','y','l','pH');
x0=x[x.length-1];
y0=y[y.length-1];

for (i=x.length-2;i>0;i-=2)
 { 
 	print (x[i], y[i], l, getPixel(x[i],y[i]));

 	// Update length and last position
 	l += sqrt((x[i]-x0)*(x[i]-x0) + (y[i]-y0)*(y[i]-y0));
 	x0 =  x[i];
 	y0 = y[i];
 }
//polygon = roi.getInterpolatedPolygon(1.0, true);
