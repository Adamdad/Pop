function stats = cvprRateImages(det1, det2, gtruthall)
% stats = cvprRateImages(det1, det2, gtruth)
% det1{ntype} and det2 are the original and final detections
% gtruth(nimages)

ntypes = 2;

gtruth{1} = gtruthall;
gtruth{2} = gtruthall;

for f = 1:numel(gtruthall)
    ind1 = find(gtruth{1}(f).objType==1);
    ind2 = find(gtruth{1}(f).objType==2);
    gtruth{1}(f).objType(ind2) = [];
    gtruth{1}(f).bbox(ind2, :) = [];
    gtruth{2}(f).objType(ind1) = [];
    gtruth{2}(f).bbox(ind1, :) = [];   
end

for t = 1:ntypes
        
    imnum1 = [det1{t}(:).imgnum];
    imnum2 = [det2{t}(:).imgnum];
    
    for f = 1:numel(gtruth{t})
        
        if mod(f, 50)==0
            disp(num2str(f))
        end
        
        ind1 = find(imnum1==f);
        ind2 = find(imnum2==f);        
    
        for k = ind1
            det1{t}(k).imgnum = 1;
        end
        for k = ind2
            det2{t}(k).imgnum = 1;
        end
        
        stats(f).valid(t) = ~isempty(gtruth{t}(f).objType);
        
        if stats(f).valid(t)        
            [stats(f).roc1(t), stats(f).pr1(t)] = tod_roc(gtruth{t}(f), det1{t}(ind1));
            [stats(f).roc2(t), stats(f).pr2(t)] = tod_roc(gtruth{t}(f), det2{t}(ind2));
        end
                        
    end
    
end
   

cthresh1 = [0.109 0.103]; % thresholds for FP/image of 2
cthresh2 = [0.276 0.163];

for f = 1:numel(stats)
%    stats(f).auc1 = mean(areaUnderCurve(stats(f).roc1));

    if mod(f, 50)==0
        disp(num2str(f))
    end

    stats(f).avep1 = averageP(stats(f).pr1);
    stats(f).avep2 = averageP(stats(f).pr2);
    if sum(stats(f).valid)>0
        stats(f).apscore1 = sum(stats(f).avep1) / sum(stats(f).valid);
        stats(f).apscore2 = sum(stats(f).avep2) / sum(stats(f).valid);       
    else
        stats(f).apscore1 = 0;
        stats(f).apscore2 = 0;
    end
    
    for t = 1:ntypes
        stats(f).pscore1(t) = 0;
        stats(f).rscore1(t) = 0;
        stats(f).pscore2(t) = 0;
        stats(f).rscore2(t) = 0;
        if stats(f).valid(t)
            ind1 = max(find(stats(f).pr1(t).conf>cthresh1(t)));
            if ~isempty(ind1)
                stats(f).pscore1(t) = stats(f).pr1(t).p(ind1);
                stats(f).rscore1(t) = stats(f).pr1(t).r(ind1);
            end
            ind2 = max(find(stats(f).pr2(t).conf>cthresh2(t)));
            if ~isempty(ind2)
                stats(f).pscore2(t) = stats(f).pr2(t).p(ind2); 
                stats(f).rscore2(t) = stats(f).pr2(t).r(ind2); 
            end
        end
        if sum(stats(f).valid)>0
            stats(f).pscore1 = sum(stats(f).pscore1) / sum(stats(f).valid);
            stats(f).pscore2 = sum(stats(f).pscore2) / sum(stats(f).valid);
            stats(f).rscore1 = sum(stats(f).rscore1) / sum(stats(f).valid);
            stats(f).rscore2 = sum(stats(f).rscore2) / sum(stats(f).valid);            
        else
            stats(f).pscore1 = 0;
            stats(f).pscore2 = 0;
            stats(f).rscore1 = 0;
            stats(f).rscore2 = 0;
        end
    end
end



% compute average precision

function p = averageP(pr)

p = zeros(size(pr));

for t = 1:length(pr)

    tp = 0;
    
    if pr(t).nobj > 0
            
        rthresh = [0:0.1:1];

        tp = zeros(size(rthresh));

        for k = 1:numel(rthresh)
            ind = find(pr(t).r > rthresh(k));
            if ~isempty(ind)
                tp(k) = max(pr(t).p(ind));
            end
        end
    end
    
    p(t) = mean(tp);
end

    

% compute area under curve

% function auc = areaUnderCurve(roc)
% 
% for t = 1:numel(roc)
% 
%     area = 0;
%     for k = 1:numel(roc(t).conf)-1
%         area = area + mean(roc(t).tp([k k+1]))*(roc(t).fp(k+1)-roc(t).
    