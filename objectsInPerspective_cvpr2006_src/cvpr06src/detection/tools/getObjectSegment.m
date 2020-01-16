function seg = getObjectSegment(img, xv, yv)
%
% returns a binary image with the mask of the object

[n,m,c] = size(img);
[x,y] = meshgrid(1:m,1:n);

seg = inpolygon(x,y, xv,yv);

