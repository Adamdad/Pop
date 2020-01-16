function [img, box, scaling, bounds] = scaleImage(img, box, nCols, nRows, patchSize, possibleScales)
% Centers the image with respect to the box(1) and scales it so the
% object width = patchSize(1). Then it crops the image so that
% the output is not larger than [nCols, nRows]

[nrows ncols ncolors] = size(img);
%maxSide = max([box(2,1)-box(1,1) box(4,1)-box(3,1)]);
%scaling = max([patchSize(1)/maxSide 64/nrows]);

%scaling = min(patchSize(1) / (box(2,1)-box(1,1)), ...
%    patchSize(2) / (box(4,1)-box(3,1)));
scaling = patchSize(1) / (box(2)-box(1));

[tmp, ind] = min(abs(scaling-possibleScales));
disp(num2str([scaling possibleScales(ind)]))
scaling = possibleScales(ind);
%disp(num2str(box'))
%disp(num2str(size(img)))
%box = box*scaling;
% box = box * scaling - 2; DWH 05/2005
cx = round((box(2)+box(1))/2 * scaling); 
cy = round((box(3)+box(4))/2 * scaling);

xmin = max([1 cx-nCols/2]);
ymin = max([1 cy-nRows/2]);

box = [cx-patchSize(1)/2+1  cx+patchSize(1)/2  cy-patchSize(2)/2+1  cy+patchSize(2)/2];

box(1:2) = box(1:2) - xmin + 1;
box(3:4) = box(3:4) - ymin + 1;

img = imresize(img, scaling, 'bilinear', 41);
%img = img(3:end-2, 3:end-2, :); DWH 05/2005
[nrows ncols ncolors] = size(img);
%disp(num2str([scaling nrows ncols]))

bounds = [ymin  min([ymin+nRows-1 nrows])  xmin  min([xmin+nCols-1 ncols])];
%bounds = [2+ymin  min([ymin+nRows-1 nrows])+2  2+xmin  min([xmin+nCols-1 ncols])+2]; DWH 05/2005

img = img(ymin:min([ymin+nRows-1 nrows]), xmin:min([xmin+nCols-1 ncols]), :);

