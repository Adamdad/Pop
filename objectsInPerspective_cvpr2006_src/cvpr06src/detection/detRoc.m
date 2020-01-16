function [roc, pr] = detRoc(gtruth, det, missing_some)
% [roc, pr] = tod_roc(gtruth, det, missing_some)
% gtruth - the ground truth detections
% det - the detection results
% missing_some (optional) - 1 if some gtruth images were not processed,
% causes images without any detections to be ignored


valid = ones(1, length(gtruth));
if exist('missing_some') && missing_some==1
    valid(setdiff([1:length(gtruth)], [det(:).imgnum])) = 0;
end

[tmp, si] = sort(-[det(:).confidence]);
det = det(si);

used = cell(length(gtruth), 1);
nobj = 0;
for i = 1:length(gtruth)
    if valid(i)        
        used{i} = zeros(size(gtruth(i).bbox, 1), 1);
        nobj = nobj + length(used{i});
    end
end

tp =zeros(length(det), 1);
for i = 1:length(det)
    imnum = det(i).imgnum;
    box = det(i).bbox; % x1 x2 y1 y2
    %det(i)
    jmax = 0;
    ovmax = -1;
    for j = 1:length(used{imnum})
        %j
        if ~used{imnum}(j)
            boxgt=gtruth(imnum).bbox(j, :); % x1 x2 y1 y2

            bi=[max(box(1),boxgt(1)) ; max(box(3),boxgt(3)) ; min(box(2),boxgt(2)) ; min(box(4),boxgt(4))]; 
            
            iw=bi(3)-bi(1)+1;
            ih=bi(4)-bi(2)+1;
            %disp([iw ih])
            if iw>0 & ih>0                
                %ua=(bu(3)-bu(1)+1)*(bu(4)-bu(2)+1);
                
                a1 = (boxgt(2)-boxgt(1)+1)*(boxgt(4)-boxgt(3)+1);
                a2 = (box(2)-box(1)+1)*(box(4)-box(3)+1);                
                ua = a1 + a2 - iw*ih;
                ov=iw*ih/ua;
                if ov>ovmax
                    ovmax=ov;
                    jmax=j;
                end
            end
        end
    end
    if jmax > 0 & ovmax > 0.33
        used{imnum}(jmax) = 1;
        tp(i) = 1;
    end    
                 
end

if 0
fpind = find(tp==0);
fpind = fpind(1:100);
fpimnum = [det(fpind).imgnum];
tpind = find(tp==1);
length(tpind)
tpind = tpind(end-24:end);
tpimnum = [det(tpind).imgnum];
imnum = intersect(fpimnum, tpimnum);
for j = 1:length(imnum)
    disp([gtruth(imnum(j)).imname])
    im = imread(['../objdet/data/images/cars/val/' gtruth(imnum(j)).imname]);
    figure(4), hold off, imshow(im);
    fpindj = fpind(find(fpimnum==imnum(j)));        
    for k = 1:length(fpindj)
        try 
            box = round(det(fpindj(k)).bbox);
            figure(4), hold on, plot(box([1 1 2 2 1]), box([3 4 4 3 3]), 'r');
            figure(5) %hold on, subplot(1, length(fpindj), j)
            imshow(imresize(rgb2gray(im(box(3):box(4), box(1):box(2), :)), [20 32], 'bilinear'))
        catch
        end
    end
    tpindj = tpind(find(tpimnum==imnum(j)));      
    for k = 1:length(tpindj)
        try
            box = round(det(tpindj(k)).bbox);
            figure(4), hold on, plot(box([1 1 2 2 1]), box([3 4 4 3 3]), 'g');
            figure(6) % hold on, subplot(1, length(tpindj), j)
            imshow(imresize(rgb2gray(im(box(3):box(4), box(1):box(2), :)), [20 32], 'bilinear'))
        catch
        end
    end 
    j
    pause
end
end
%for i = 1:20
%    for j = 1:length(used{i})
%        if used{i}(j)

fp = 1-tp;

roc.tp = cumsum(tp) / nobj;
roc.fp = cumsum(fp);
roc.fp = roc.fp / sum(valid);
roc.conf = [det(:).confidence]';
roc.nobj = nobj;
roc.nimages = sum(valid);
roc.tested = find(valid);

pr.p = cumsum(tp) ./ cumsum(tp+fp);
pr.r = cumsum(tp) / nobj;
pr.ndet = length(tp);
pr.nimages = sum(valid);
pr.nobj = nobj;
pr.conf = [det(:).confidence]';
pr.tested = find(valid);
        
