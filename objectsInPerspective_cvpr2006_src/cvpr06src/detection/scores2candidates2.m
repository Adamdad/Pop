function candidates = scores2candidates2(scores, sparam, objsize, maxOv, minConf, maxdet)
% candidates = scores2candidates(scores, sparam, objsize, maxOv, minConf, maxdet)
% Computes a set of candidate objects from object detection scores.
% Candidates consist of a possible type and a point distribution of
% locations.  Uses prune_detections to find candidate centers.
%
% Input:
%   scores{nscales,nsize}(ny, nx): confidences of detections at each pixel
%           and scale
%   sparam: the scaling factor (scores{s}(i, -) is has a y-value (row) of
%           i/sparam^(s-1) in the image)
%   objsize(nsize, [w h]): the size of an object window at original scale
%   maxOv: maximum overlap between boxes
%   minConf: minimum confidence for a detection
%   maxdet: maximum number of detections
% Output:
%   candidates(ncand).(bbox, conf, p):
%       bbox(i, [x1 x2 y1 y2]): the bounding box of detection i
%       conf(i): the confidence of the object being at i given that the
%                candidate is the object
%       p: the overall confidence for the candidate

candidates = [];

if ~exist('maxdet')
    maxdet = Inf;
end

% get bounding boxes
nsize = size(objsize, 1);
det = [];
for s = 1:nsize
    det = [det ; scores2detections(scores(:, s), objsize(s, :), sparam, minConf)];
end
det = -sortrows(-det, 1);

tmpdet = output2det({det(:, 1)}, {det(:, 2:5)}, 1);

tmpdet = prune_detections(tmpdet, maxOv, minConf, maxdet);
det2 = zeros(length(tmpdet), 5);
for j = 1:length(tmpdet)
    det2(j, :) = [tmpdet(j).confidence tmpdet(j).bbox];
end
det2 = -sortrows(-det2, 1);
ncand = size(det2, 1);

% get candidate membership for each detection (closest in sense of maximum
% overlap = area(A AND B) / area(A OR B) )
ind = zeros(size(det, 1), 1);
for j = 1:size(det, 1)
    ov = detComputeOverlap(det(j, 2:5), det2(:, 2:5));
    [tmp, ind(j)] = max(ov);
end

for c = 1:ncand
    cind = find(ind==c);
    candidates(c).conf = det(cind, 1);
    candidates(c).bbox = det(cind, 2:5);    
    candidates(c).p = max(candidates(c).conf);
end


