function candidates = det2candidates(det, maxov)
% candidates = det2candidates2(det, maxov)
% Computes a set of candidate objects from object detection bounding boxes
% and confidences.  Candidates consist of a possible type and a point 
% distribution of locations.  Uses prune_detections to find candidate 
% centers.
%
% Input:
%   det(ndet, [conf x1 x2 y1 y2]): detections
%   maxov: maximum overlap between boxes
% Output:
%   candidates(ncand).(bbox, conf, p):
%       bbox(i, [x1 x2 y1 y2]): the bounding box of detection i
%       conf(i): the confidence of the object being at i given that the
%                candidate is the object; confidences should be marginal
%                probabilities
%       p: the overall confidence for the candidate

candidates = [];

% get set of candidate centers (det2)
tmpdet = output2det({det(:, 1)}, {det(:, 2:5)}, 1);
tmpdet = detPruneDetections(tmpdet, maxov, -Inf, Inf);
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


