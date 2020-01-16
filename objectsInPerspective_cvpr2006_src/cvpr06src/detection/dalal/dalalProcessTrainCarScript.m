% dalalProcessTrainCarScript

%objsize = [88 32 ; 64 32 ; 40 32];
objsize = [80 24];

%load '/IUS/vmr7/dhoiem/context/objdet/data/car_groundtruth.mat'
load '/IUS/vmr7/dhoiem/context/objdet/data/dalal_car_gtruth2.mat'

basedir = '/IUS/vmr7/dhoiem/context/objdet/data/images/cars';
posdir = '/IUS/vmr20/dhoiem/datasets/dalal/cars/OLT/train/pos';
negdir = '/IUS/vmr20/dhoiem/datasets/dalal/cars/OLT/train/neg';

% imdir{1} = [basedir];
% imdir{2} = [basedir];
% imdir{3} = [basedir '/test1'];
% imdir{4} = [basedir '/test2'];

% gt{1} = gtruthtrain;
% gt{2} = gtruthval;
% gt{3} = gtruthtest1;
% gt{4} = gtruthtest2;

pcount = 0;
ncount = 0;   

disp('not copying negative images')

for f = 1:numel(gtruth)

    infn = [basedir '/' gtruth(f).imname];        
    [bn, ext] = strtok(gtruth(f).imname, '.');

    if isempty(gtruth(f).bbox)

        ncount = ncount + 1;

        nstr = num2str(10000+ncount);
        outfn = [negdir '/carneg_' nstr(2:end) ext];
        %system(['cp ' infn ' ' outfn]);

    else

        im = imread(infn);

        for b = 1:size(gtruth(f).bbox, 1)

            bbox = gtruth(f).bbox(b, :);                
            h = bbox(4)-bbox(3)+1;
            w = bbox(2)-bbox(1)+1;
            hwrat = w/h;
            stdrat = objsize(:, 1)./objsize(:, 2);

            [tmp, os] = min(abs(stdrat-hwrat));

            if h >= objsize(os, 2) && ...
                    bbox(1)>1 && bbox(2) < size(im, 2) && bbox(3)>1 && bbox(4)<size(im, 1)

                pcount = pcount + 1;

                % rescale and center image on object
                scale = objsize(os, 2)/h;

                im2 = imresize(im, scale, 'bicubic');
                im2 = padarray(im2, [96 96 0], 'symmetric');
                imcenter = size(im2)/2;
                bbox = round(bbox*scale+96);
%                    bbox = round([imcenter(2) + [-objsize(1)/2+0.5 objsize(1)/2-0.5] ...
%                        imcenter(1) + [-objsize(2)/2+0.5 objsize(2)/2-0.5]]);

                im3 = im2(bbox(3)-32:bbox(4)+32, bbox(1)-32:bbox(2)+32, :);

                nstr = num2str(10000+pcount);                    
                outfn = [posdir '/carpos_' num2str(os) '_' nstr(2:end) ext];
                imwrite(im3, outfn);
                
                if mod(pcount, 50)==0
                    disp(num2str(pcount))
                end
            end
        end
    end
end


                    
                
                