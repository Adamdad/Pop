function [det2, keepind] = detPruneDetections(det, maxOverlap, minConf, maxdet)
% det2 = detPruneDetections(det, maxOverlap, minConf, maxdet)

if ~exist('maxdet')
    maxdet = Inf;
end

%ind = find([det(:).origconf]<minConf);
ind = find([det(:).confidence]<minConf);
det(ind) = [];

imnum = [det(:).imgnum];
maxim = max(imnum);
minim = min(imnum);

det2 = det;
count = 0;

keepind = [];

for f = minim:maxim
    ind = find(imnum==f);
    fdet = det(ind);
    
    % sort by confidence, lowest first
    %[tmp, si] = sort([fdet(:).origconf]);
    [tmp, si] = sort([fdet(:).confidence]);            
    
    firstind = max(1, numel(si)-maxdet+1);
    
    fdet = fdet(si(firstind:end));
        
    if ~mod(f, 100)
        disp(num2str(f))
    end
    
    keep = ones(length(fdet), 1);
    
    for i = 1:length(fdet)
        box = fdet(i).bbox;                
        
        for j = i+1:length(fdet)
            
            boxgt = fdet(j).bbox;
             
            bi=[max(box(1),boxgt(1)) ; max(box(3),boxgt(3)) ; min(box(2),boxgt(2)) ; min(box(4),boxgt(4))];

            iw=bi(3)-bi(1)+1;
            ih=bi(4)-bi(2)+1;

            if iw>0 && ih>0 
                
                a1 = (boxgt(2)-boxgt(1)+1)*(boxgt(4)-boxgt(3)+1);
                a2 = (box(2)-box(1)+1)*(box(4)-box(3)+1);    
                           
                ua = a1 + a2 - iw*ih;        
                ov=iw*ih/ua;
                if ov>maxOverlap
                    keep(i) = 0;
                    break;                   
                end                                
            end                 
        end
    end

    fdet = fdet(find(keep));
    det2(count+1:count+length(fdet)) = fdet;
    count = count + length(fdet);
    
    if nargout > 1
        keepind = [keepind ; ind(find(keep))];
    end
end

det2 = det2(1:count);

