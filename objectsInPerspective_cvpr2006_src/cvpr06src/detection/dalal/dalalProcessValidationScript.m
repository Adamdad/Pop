% process dalal validation script

DO_PEOPLE = 1;
DO_CARS = 0;

load '/IUS/vmr20/dhoiem/data/valset.mat'

% PEOPLE

dalal_margin = [8 8]; % normally [16 16]

if DO_PEOPLE

    % read detections from files and put in order of ground truth
    fn = '/IUS/vmr20/dhoiem/datasets/dalal/OLT/valresults/out_sm.txt';
    listfn = '/IUS/vmr20/dhoiem/datasets/dalal/OLT/vallist.txt';
    [valdet, vallist] = dalalOutput2Detections(fn, listfn, dalal_margin);
    [valdet, vallist] = detCombineDetections(cell(size(vs.imnames)), vs.imnames, valdet, vallist);

    % convert detection format to standard used by pruning and roc functions
    for f = 1:numel(valdet)
        conf{f} = valdet{f}(:, 1);
        bbox{f} = valdet{f}(:, 2:5);
    end
    det = output2det(conf, bbox, [1:numel(valdet)]');
    det2 = detPruneDetections(det, 0.33, -Inf);

    % convert ground truth so that each cell contains ground truth for only one
    % object type and compute roc
    gt = detSplitGroundTruthByObjects(vs.gtruth, 2);
    rocppl = detRoc(gt, det2);

    % find probabilistic output parameters
    tp = [0 ; rocppl.tp];
    tp = tp(2:end)>tp(1:end-1);
    [pplA, pplB, err] = getProbabilisticOutputParams(rocppl.conf, tp)

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% CARS

if DO_CARS

    for s = 1:1 % three shapes
    
        sstr = num2str(s);
        
        % read detections from files and put in order of ground truth
%        fn = ['/IUS/vmr20/dhoiem/datasets/dalal/cars/OLT/valresults/out' sstr '.txt'];
        fn = ['/IUS/vmr20/dhoiem/datasets/dalal/cars/OLT/valresults/out.txt'];
        listfn = '/IUS/vmr20/dhoiem/datasets/dalal/cars/OLT/vallist.txt';
        [valdet, vallist] = dalalOutput2Detections(fn, listfn, dalal_margin);
        [valdet, vallist] = detCombineDetections(cell(size(vs.imnames)), vs.imnames, valdet, vallist);

        % convert detection format to standard used by pruning and roc functions
        for f = 1:numel(valdet)
            conf{f} = valdet{f}(:, 1);
            bbox{f} = valdet{f}(:, 2:5);
        end
        det = output2det(conf, bbox, [1:numel(valdet)]');
        det2 = detPruneDetections(det, 0.33, -Inf);

        % convert ground truth so that each cell contains ground truth for only one
        % object type and compute roc
        gt = detSplitGroundTruthByObjects(vs.gtruth, 1);
        roccar(s) = detRoc(gt, det2);

        % find probabilistic output parameters
        tp = [0 ; roccar(s).tp];
        tp = tp(2:end)>tp(1:end-1);
        [carA(s), carB(s), err(s)] = getProbabilisticOutputParams(roccar(s).conf, tp)

    end
        
end