function [gtruth, objset] = detInitData(imdir, objname, datatype, outdir, ...
                                     doreshape, objsizes, patchdir)
% [gtruth, objset] = initData(imdir, objname, datatype, outdir, 
%                             doreshape, objsizes)
%
% Input:
% imdir - the directory where the images are stored
% objname - the name of the object class (e.g. VOCpeople or VOCcars)
% datatype{ntypes} - the name of the set (e.g. train or val or test1)
% outdir - where to write the images
% doreshape - 1 if the ground truth box should be reshaped according to the
%             cannonical bounding box aspect ratio (objsizes)
% obsizes(nratios, [w h]) - bounding box ratio; if nratios > 1, choose the 
%                           best fitting box

VOCinit;

if ~iscell(datatype)
    datatype = {datatype};
end

gtruth = repmat(struct('imname', '', 'bbox', [], 'boxnum', 0), 0, 1);

% for each object set
for dti = 1:length(datatype)
      
    % objname was 'VOCcars'
    objset(dti) = VOCreadimgset(PASopts, objname, datatype{dti});
    % bbox(nobj, x1 x2 y1 y2)
    % for training: 1 < boxnum(nobj) < nratios, otherwise boxnum = 0


    ratios = objsizes(:, 1) ./ objsizes(:, 2);

    used = zeros(size(objset(dti).recs));

    poscount = 0;

    for f = 1:length(objset(dti).recs)
        imname = objset(dti).recs(f).imgname;                   
        im = imread([imdir '/' imname]);

        % only use color images
        if size(im, 3)==3 

            used(f) = 1;

            s = findstr(imname, '/');
            imname = [datatype{dti} '/' imname(s(end)+1:end)];
            while exist([outdir '/' imname])
                disp(['warning: ' imname ' exists; changing name']);
                imname = [strtok(imname, '.') '-.png'];
                disp(['new name = ' imname])
            end
            disp([outdir '/' imname])        
            gtruth(end+1, 1).imname = imname;      
            gtruth(end, 1).color = (size(im, 3)==3);        
            
            if ~isempty(outdir)
                imwrite(im, [outdir '/' imname]);
            end
            %disp('warning: not writing full file')
            %imwrite([1], [outdir '/' imname]);
            
            for j = 1:length(objset(dti).recs(f).objects)

                used(f) = 2;

                box = objset(dti).recs(f).objects(j).bbox([1 3 2 4]);

                if doreshape
                    % resize box to be in proportions of objsize                        
                    w = box(2)-box(1)+1;
                    h = box(4)-box(3)+1;

                    [tmp, boxnum] = min(abs(ratios-(w/h)));
                    gtruth(end, 1).boxnum(end+1, 1) = boxnum;
                    objsize = objsizes(boxnum, :);

                    w = h*objsize(1)/objsize(2);  % box determined by object height
    %                 if h/w < objsize(2)/objsize(1)
    %                     h = w/objsize(1)*objsize(2);                
    %                 elseif w/h < objsize(1)/objsize(2)
    %                     w = h/objsize(2)*objsize(1);
    %                 end
                    x = mean(box(1:2));
                    y = mean(box(3:4));
                    box = [x-(w-1)/2 x+(w-1)/2 y-(h-1)/2 y+(h-1)/2];

                    try 
                        if exist('patchdir') && ~isempty(patchdir)
                            patchname = [patchdir '/obj' num2str(boxnum) ...
                                '_' num2str(length(gtruth)) '_' ...
                                num2str(size(gtruth(end, 1).bbox, 1)+1) '.jpg'];
                            imwrite(imresize(im(round(box(3):box(4)), ...
                                round(box(1):box(2)), :), [objsize(2) objsize(1)]), patchname);
                        end
                    catch; end;
                end
                gtruth(end, 1).bbox(end+1, 1:4) = box;
            end
        end
    end

    ind = find(used>0);
    objset(dti).recs = objset(dti).recs(ind);
    objset(dti).posinds = find(used(ind)==2);
    objset(dti).neginds = find(used(ind)==1);
end