function candidates = detReadAllCandidates(fn, objtypes, sources, ovthresh, minprob, maxdet)
% candidates = detReadAllCandidates(fn, objtypes, sources)
% Reads all candidates from specified sources
%
% Input:
%   fn: cell array of image filenames
%   objtypes: the types of objects to read
%   sources: cell array denoting set of sources and corresponding options
%        The following are acceptable names/formats for sources:
%        1)  {'mtf', {objtype 1 dir, ..., objtype n dir}, sigmoidParams, 
%             szranges}: MurphyTorralbaFreeman-based 
%        2)  {'dalal', {objtype 1 outfile, ... , objtype n outfile}, 
%             {objtype 1 list, ... , objtype n list}, sigmoidParams, 
%             szranges, margins}: Dalal-Triggs
%     sigmoidParams(ntypes, [A B]) is the set of parameters for converting
%       from the confidence score to the probability
%     szranges(ntypes, [minw maxw minh maxh] is the range of acceptable 
%       bounding box widths and heights (in pixels)   
%   ovthresh: the overlap threshold for merging candidates
%   minprob: minimum probability for a detection to be included
%   maxdet: maximum detections per type per image
%
% Notes: a) for mtf, scale step is assumed to be 0.89


% read all detections
det = cell(numel(fn), numel(objtypes));

disp('Reading detections.')

for n = 1:numel(objtypes)

    disp(['Type: ' num2str(objtypes(n))]);
    
    for src = 1:numel(sources)
        srcname = sources{src}{1};

        disp(['Source: ' srcname]);

        srcdet = [];
        if strcmp(lower(srcname), 'mtf')
            if ~isempty(sources{src}{2}{n})
                srcdet = readDetectionsMTF(fn, objtypes(n), ...
                    sources{src}{2}{n}, sources{src}{3}(n, :), ...
                    sources{src}{4}(n, :), minprob, maxdet);
            end
        elseif strcmp(lower(srcname), 'dalal')
            if ~isempty(sources{src}{2}{n})            
                srcdet = readDetectionsDalal(fn, sources{src}{2}{n}, ...
                    sources{src}{3}{n}, sources{src}{4}(n, :), ...
                    sources{src}{5}(n, :), minprob, maxdet, sources{src}{6}(n));
            end
        else
            error(['invalid source name: ' srcname])
        end              
        
        if ~isempty(srcdet)
            for f = 1:size(det, 1)
                det{f, n} = [det{f, n} ; srcdet{f}];
            end        
        end
    end        
    
    for f = 1:size(det, 1)
        if size(det{f, n},1)>maxdet
            det{f, n} = -sortrows(-det{f, n}, 1);
            det{f, n} = det{f, n}(1:maxdet, :);
        end
    end
    
end


% convert dets to candidates
disp('Converting to candidates.')
candidates = cell(numel(fn), 1);
for f = 1:numel(candidates)
    candidates{f} = [];    
    for t = 1:numel(objtypes)
        tmpcand = det2candidates(det{f, t}, ovthresh);
        for k = 1:numel(tmpcand)
            tmpcand(k).objType = objtypes(t);
        end
        candidates{f} = [candidates{f} tmpcand];
    end            
end
    
% end detReadAllCandidates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function det = readDetectionsMTF(fn, objtype, folder, sigAB, szrange, minprob, maxdet)

det = cell(size(fn));

ots = num2str(objtype);

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
                scores{sc, sz}(objdet(sz).ind{sc}) = ...
                    1./(1+exp(sigAB(1)*objdet(sz).scores{sc}+sigAB(2)));
            end              
        end
        sparam = 0.89;
        objsize = reshape([objdet(:).objsize], 2, numel(objdet))'; 
        det{f} = scores2detections(scores, objsize, sparam, ...
            'minscore', minprob, 'maxdets', Inf);   
        
        % remove boxes that are too small or too big (according to szrange)        
        deth = det{f}(:, 5)-det{f}(: ,4)+1;
        ind = find((deth < szrange(3)) | (deth > szrange(4)));
        det{f}(ind, :) = [];
        detw = det{f}(:, 3)-det{f}(:, 2)+1;
        ind = find((detw < szrange(1)) | (detw > szrange(2)));
        det{f}(ind, :) = [];     
        
        det{f} = -sortrows(-det{f}, 1);
        if size(det{f}, 1) > maxdet
            det{f} = det{f}(1:maxdet, :);
        end        
        
    else
        det{f} = [];
        disp(['warning: ' loadName ' not found'])
    end
end
  
% end readDetectionsMTF
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function det = readDetectionsDalal(fn, outfn, listfn, sigAB, szrange, minprob, maxdet, margin)

% read detections and index according to fn
[tmpdet, list] = dalalOutput2Detections(outfn, listfn, [margin margin]);
det = detCombineDetections(cell(size(fn)), fn, tmpdet, list);

% convert scores to probabilities and remove dets with low scores
for f = 1:numel(fn)
    det{f}(:, 1) = 1 ./ (1+exp(sigAB(1)*det{f}(:, 1) + sigAB(2)));
    det{f} = det{f}(find(det{f}(:, 1)>=minprob), :);
    
    % remove boxes that are too small or too big (according to szrange)        
    deth = det{f}(:, 5)-det{f}(: ,4)+1;
    ind = find((deth < szrange(3)) | (deth > szrange(4)));
    det{f}(ind, :) = [];
    detw = det{f}(:, 3)-det{f}(:, 2)+1;
    ind = find((detw < szrange(1)) | (detw > szrange(2)));
    det{f}(ind, :) = [];              
        
    det{f} = -sortrows(-det{f}, 1);
    if size(det{f}, 1) > maxdet
        det{f} = det{f}(1:maxdet, :);
    end
end

% end readDetectionsDalal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



        
