function [priorY, yc] = viewTrainCameraHeightPrior(labeledHorizons, gtruth, gtnames)
% Estimates the prior for the camera height given object bounding boxes and
% labeled horizon positions.
% labeledHorizons - "imname" the name of the image
%                   "valid" true if the horizon is labeled
%                   "b" the pixel height of horizon in the image (from top)                   
%                   "height" - the height of the image
% gtruth(nimages) - ".bbox(nobj, [x1 x2 y1 y2])", 
%                   ".objType"
% gtnames(nimages) - image name for each ground truth
% objrecs - "imgname" the name of the image
%           "objects(nobj).bbox" [x1 y1 x2 y2]

% car size model in meters
models(1).mu = 1.59;
models(1).sigma = 0.21;

% pedestrian (adult)
models(2).mu = 1.7;
models(2).sigma = 0.085;

        
yc = [];

for i = 1:length(labeledHorizons)

    imgind = find(strcmp(gtnames, labeledHorizons(i).imname));

    if isempty(imgind) || length(gtruth(imgind).objType)<=0
        disp(i)
    end
    
    if length(imgind)>0  && length(gtruth(imgind).objType)>0 && labeledHorizons(i).valid
               
        imh = labeledHorizons(i).height;
        bbox = gtruth(imgind).bbox;

        % get normalized horizon position
        % labeledHorizon(i).b is measured from top to bottom as 1-height
        % pixels, but we need it measured from bottom to top as 0-1       
        v0 = (imh-labeledHorizons(i).b+1)/imh;        
        
        h = []; v = []; t = [];
        
        % get object identities and normalized heights and positions
        t = gtruth(imgind).objType;
        h = (bbox(:, 4)-bbox(:, 3)+1)/imh;
        v = (imh-bbox(:, 4)+1)/imh;
        

        yc(i) = cameraHeightML(v0, v, h, t, models);
                
    else
        yc(i) = nan;
    end
end

ind = find(~isnan(yc));

n = length(yc(ind));
sorty = sort(yc(ind));
iq = sorty(round(3/4*n))-sorty(round(n/4));
kernelwidth = 0.9*min(std(yc(ind)), iq/1.34)*(1/n)^(1/5); 

[priorY.f, priorY.x] = ksdensity(yc(ind), 'width', kernelwidth, 'npoints', 50);

