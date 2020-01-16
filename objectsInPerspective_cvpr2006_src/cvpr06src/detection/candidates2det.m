function [det, det2] = candidates2det(candidates, objtype, ind2imnum)
% [det, det2] = candidates2det(candidates, objtype, ind2imnum)
% 
% Converts the candidate structure to the det structure
% candidates{nimages}.bbox(nbox, [x1 x2 y1 y2])
%                    .conf(nbox)
%                    .p
% det{ntype}(nbox).(imgnum, confidence, bbox(x1 x2 y1 y2))
% ind2imnum: gives the image number for each index of candidates (default
% is image number equals candidate index)

cthresh = 0.0;

if ~exist('ind2imnum')
    ind2imnum = [1:numel(candidates)];
end

for t = objtype

    nboxes = 0;
    for f = 1:length(candidates)
        for k = 1:numel(candidates{f})
            if candidates{f}(k).objType==t
                pobj = candidates{f}(k).p;
                maxc = max(candidates{f}(k).conf);
                candidates{f}(k).conf = candidates{f}(k).conf / (maxc+1E-10) * pobj;                        
                nboxes = nboxes + sum(candidates{f}(k).conf>cthresh);
            end
        end
    end

    det{t} = repmat(struct('imgnum', 0, 'confidence', 0, 'bbox', zeros(1, 4)), 1, nboxes);
    %disp(['ndet = ' num2str(nboxes)])

    cnt = 0;

    for f = 1:length(candidates)    

        imnum = ind2imnum(f);

        if mod(f, 100)==0
            disp(num2str(f))
        end

        for k = 1:numel(candidates{f})

            if candidates{f}(k).objType==t
                for j = find(candidates{f}(k).conf>cthresh)'%1:numel(candidates{f}(k).conf)            
                    cnt = cnt + 1;
                    det{t}(cnt).imgnum = imnum;
                    det{t}(cnt).confidence = candidates{f}(k).conf(j);
                    det{t}(cnt).bbox(1:4) = candidates{f}(k).bbox(j, :);
                end
            end
        end
    end

    det2{t} = detPruneDetections(det{t}, 0.34, 0.0001);

end
