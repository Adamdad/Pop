function gt = detSplitGroundTruthByObjects(gtruth, objtypes)

gt = repmat(struct('bbox', []), length(gtruth), length(objtypes));
for t = 1:length(objtypes)
    for f = 1:length(gtruth)    
        ind = find(gtruth(f).objType==objtypes(t));
        gt(f,t).bbox = gtruth(f).bbox(ind, :);
    end
end
