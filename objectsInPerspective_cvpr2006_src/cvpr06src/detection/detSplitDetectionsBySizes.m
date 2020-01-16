function det2 = detSplitDetectionsBySizes(det, bbsizes)
% det2 = detSplitDetectionsBySizes(det, bbsizes)
% Splits detections into structures according to bbsizes.
%
% Input:
%   det(nboxes).{bbox, conf, objType, imgnum}
%   bbsizes(nsizes, [minw maxw minh maxh])
% Output:
%   det2{nsizes}(nboxes)
%

nsizes = size(bbsizes, 1);

for s = 1:nsizes
    sz = bbsizes(s, :);
    keep = zeros(size(det));
    for b = 1:numel(det)    
        detw = det(b).bbox(2)-det(b).bbox(1)+1;
        deth = det(b).bbox(4)-det(b).bbox(3)+1;
        keep(b) = (detw >= sz(1)) && (detw <= sz(2)) && ...
            (deth >= sz(3)) && (deth <= sz(4));
    end    
    det2{s} = det(find(keep));
end