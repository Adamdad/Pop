% process mtf validation script

load '/IUS/vmr20/dhoiem/data/valset.mat'
fn = vs.imnames;

folder = '/IUS/vmr20/dhoiem/datasets/valset/images/objectDetections';

bboxrange{1} = [0 Inf 0 32]; % [minw maxw minh maxh] car size range
bboxrange{2} = [0 Inf 0 96]; % [minw maxw minh maxh] ppl size range

for t = 1:2

    disp(['Type: ' num2str(t)])
    
    ots = num2str(t);
    
    for f = 1:numel(fn)        
        loadName = [folder '/' strtok(fn{f}, '.') '.objdet.' ots '.mat'];
        if exist(loadName)
            tmp = load(loadName);
            objdet = tmp.passeddet;
        
            % scores{nshapes, nscales}
            scores = cell(numel(objdet), size(objdet(1).size, 1));
            for sz = 1:length(objdet)
                for sc = 1:size(objdet(sz).size, 1)                                                
                    scores{sc, sz} = repmat(-Inf, objdet(sz).size(sc, :));                         
                    scores{sc, sz}(objdet(sz).ind{sc}) = objdet(sz).scores{sc};
                end              
            end
        end
        sparam = 0.89;
        objsize = reshape([objdet(:).objsize], 2, numel(objdet))'; 
        valdet{f} = scores2detections(scores, objsize, sparam, ...
            'minscore', 0, 'maxdets', 1000);  
    end

    for f = 1:numel(valdet)
        conf{f} = valdet{f}(:, 1);
        bbox{f} = valdet{f}(:, 2:5);
    end
    det = output2det(conf, bbox, [1:numel(valdet)]');
    det2 = detPruneDetections(det, 0.33, -Inf);    
    det2 = detSplitDetectionsBySizes(det2, bboxrange{t});
    
    gt = detSplitGroundTruthByObjects(vs.gtruth, t);
    gt = detSplitGroundTruthBySizes(gt, bboxrange{t});

    roc(t) = detRoc(gt, det2{1});
    
    % find probabilistic output parameters
    tp = [0 ; roc(t).tp];
    tp = tp(2:end)>tp(1:end-1);
    [A(t), B(t), err(t)] = getProbabilisticOutputParams(roc(t).conf, tp)    
    
end