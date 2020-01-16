function [roc, pr] = candidates2roc(gtruth, candidates, objtypes, bbsizes)
% [roc, pr] = candidates2roc(gtruth, candidates, objtypes, bbsizes)
% Computes roc and pr curves for object detection and localization accuracy
%
% Input:
%   Indices of candidates must correspond with indices of gtruth
%   gtruth(nimages).bbox(nboxes, [x1 x2 y1 y2]): bounding boxes
%                  .objType(nboxes): object types for each box
%   candidates{nimages}(ncand).bbox(nboxes, [x1 x2 y1 y2]): bounding boxes
%                             .conf(nboxes): confidence for each box
%                             .p: probability that candidate is object
%                             .objType: type of object for each candidate
%   objtypes(ntypes): the nums of the objects for which there are candids.
%   bbsizes{ntypes}(nsizes, [minw maxw minh max]) (optional): if included
%     this computes roc curves for detections within the range(s) specified
% Output:
%   roc(ntypes, nsizes) - roc curves for each type and size
%   pr(ntypes, nsizes) - precision-recall curves for each type and size
%

if ~exist('bbsizes', 'var')
    for t = 1:numel(objtypes)
        bbsizes{t}(1, 1:4) = [0 Inf 0 Inf];
    end
end

gt = detSplitGroundTruthByObjects(gtruth, objtypes);

[det, det2] = candidates2det(candidates, objtypes);

for t = 1:numel(objtypes)
    for s = 1:size(bbsizes{t}, 1)
        tmpgt = detSplitGroundTruthBySizes(gt(:, t), bbsizes{t}(s, :));
        tmpdet = detSplitDetectionsBySizes(det2{objtypes(t)}, bbsizes{t}(s, :));    
        [roc(t, s), pr(t, s)] = detRoc(tmpgt, tmpdet{1});    
    end
end
    