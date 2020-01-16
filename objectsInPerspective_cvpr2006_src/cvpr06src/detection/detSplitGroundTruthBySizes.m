function gt = detSplitGroundTruthBySizes(gtruth, bbsizes)
% gt = detSplitGroundTruthBySizes(gtruth, bbsizes)
% Splits ground truth into structures according to bbsizes.
%
% Input:
%   gtruth(nimages).{bbox, objType}
%   bbsizes(nsizes, [minw maxw minh maxh])
% Output:
%   gtruth(nimages, nsizes)
%

nsizes = size(bbsizes, 1);

gt = repmat(struct('bbox', []), numel(gtruth), nsizes);
for s = 1:nsizes
    sz = bbsizes(s, :);
    for f = 1:numel(gtruth)  
        if ~isempty(gtruth(f).bbox)
            gtw = gtruth(f).bbox(:, 2)-gtruth(f).bbox(:, 1)+1;
            gth = gtruth(f).bbox(:, 4)-gtruth(f).bbox(:, 3)+1;
            ind = find((gtw >= sz(1)) &  (gtw <= sz(2)) & ...
                (gth >= sz(3)) & (gth <= sz(4)));
            gt(f,s).bbox = gtruth(f).bbox(ind, :);
        end
    end
end