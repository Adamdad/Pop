function data = detCreateData(imdir, detparams, gtruth)
% CREATE LABELS DATABASE:
% This program creates a folder with images in which the target has
% normalized size. It also stores the location of the target.
%
% imdir - the location of the images              
% detparams.objectSize: the size of the object window (suggest [32 12])
%           imageMaxSize: the max extent surrounding the object patch
%                         (suggest [100 100])
% gtruth(nimages).imname: the name of the image
%                 bbox(nobjs, [y1 y2 x1 x2]): the object bounding boxes
% geomlabels: (deprecated) the ground-truth geometric labels for the object

USE_CONTEXT = 0;
USE_COLOR = 0;

% PARAMETERS OF NORMALIZED OBJECTS:
objectSize = detparams.objectSize;
imageMaxSize = detparams.imageMaxSize;

nimages = length(gtruth);
  
ind = 0;

for f = 1:nimages
    
    nobj = size(gtruth(f).bbox, 1);
    
    if nobj > 0
        disp([num2str(f) ': ' gtruth(f).imname])

        img = im2double(imread([imdir '/' gtruth(f).imname]));
        if size(img, 3)==3 && ~USE_COLOR
            img = rgb2gray(img);
        end  

    end
        
    for i = 1:nobj

        box = gtruth(f).bbox(i, :)';   
        possibleScales = detparams.scaleSampling.^([40:-1:-1]);
       
        [img2, box, scaling, bnd] = scaleImage(img, box, ...
                         imageMaxSize(2), imageMaxSize(1), ...
                         objectSize(gtruth(f).boxnum(i), :), possibleScales);             
                         
        %    scaling
        mo = 2; % max pixels out of image
        if (scaling <=1)  &&  (box(3) >= 1-mo) && ...
                (box(4) <= size(img2, 1)+mo) && (box(1) >= 1-mo) && ...
                (box(2) <= size(img2, 2)+mo)
            
%            imagesc(img2); colormap(gray(256)); drawnow

                %y1 = round(max(box(3), 1));
                %y2 = round(min(box(4), size(img2, 1)));
                %x1 = round(max(box(1), 1));
                %x2 = round(min(box(2), size(img2, 2)));
                %disp(num2str([y1 y2 x1 x2]))
                %disp(num2str(size(img2)))
                %img2 = img2(y1:y2, x1:x2, :);          
                
            %if max(box(2)-box(1), box(4)-box(3)) == max(objectSize)
            if ~isempty(img2)
                ind = ind+1;
                nameImg = ['patches/' strrep(sprintf('%3d', ind),' ','0') '.jpg'];                
                disp(nameImg)
                %disp('not writing')
                imwrite(img2, [imdir '/' nameImg], 'jpg', 'quality', 100);                               
                
                if USE_CONTEXT
                    [tok, rem] = strtok(gtruth(f).imname, '/');
                    fn = [imdir '/' tok '/context' strtok(rem, '.') '.c.mat'];
                    if exist(fn)
                        load(fn);
                        cimages{1} = imresize(cimages{1}, scaling, 'bilinear', 41);
                        cimages{1} = cimages{1}(bnd(1):bnd(2), bnd(3):bnd(4), :);                    
                        save([imdir '/' strtok(nameImg, '.') '.c.mat'], 'cimages', 'cnames');
                    else            
                        disp(['No context confidences available for ' gtruth(f).imname])
                    end
                end
                
                if ~USE_CONTEXT || exist(fn)
                    data(ind).nameImg = nameImg;
                    data(ind).x   = (box(1)+box(2))/2;
                    data(ind).y   = (box(3)+box(4))/2;
                    data(ind).box = box;
                    data(ind).boxnum = gtruth(f).boxnum(i);
                    data(ind).index = f;
                    data(ind).origbox = gtruth(f).bbox(i, :);
%                     if exist('geomlabels')
%                         data(ind).geom = geomlabels(f).geom(i, :);
%                     end
                else
                    ind = ind - 1;
                end
            end      
        end
    end
end



