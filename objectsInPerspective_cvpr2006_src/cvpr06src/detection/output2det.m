function det = output2det(conf, bbox, imnums);
% Convert cell arrays of confidences and bounding boxes to the det
% structure.  One cell array per image.

nconf = numel(conf);
ndet = 0;
for f = 1:nconf
    ndet = ndet + length(conf{f});
end

det = repmat(struct('imgnum', 0, 'confidence', 0, 'bbox', [0 0 0 0]), ndet, 1);

ndet = 0;
for f = 1:nconf    
    for k = 1:length(conf{f})
        det(ndet+k).imgnum = imnums(f);
        det(ndet+k).confidence = conf{f}(k);
        det(ndet+k).bbox = bbox{f}(k, :);
   
    end
    ndet = ndet + length(conf{f});
end