function [features, classes]  = ...
    detComputeFeaturesContext(imdir, detParams, imdata, dictionary, gtruth)
% 3) Computes the input-output convolutional features 
%
%
% Input:
% imdir - the location of the images        
% detparams.NnegativeSamplesPerPositive: (suggest 100)
% data: the data from tod_createData
% dictionary: the data from tod_createDictionary
% gtruth(nimages).imname: the name of the image
%                 bbox(nobjs, [y1 y2 x2 x1]): the object bounding boxes
% c2p.o(8).{x, f} - parameters to convert average confidences to probabilities (deprecated)                
% Output:
% features{nimages}(ndata, nfeatures): feature data
% classes{nimages}(ndata): class labels

LOAD = 0;

% PARAMETERS:
NnegativeSamplesPerPositive = detParams.NnegativeSamplesPerPositive; 
scaleSampling = detParams.scaleSampling;
maxImageSize = detParams.maxNegSize;
% Compute atures:

Nimages = length(gtruth);
Nfeatures = length(dictionary);
NcontextFeatures = 30;

nPos = length(imdata);
isneg = zeros(Nimages, 1);
for i = 1:Nimages
    if isempty(gtruth(i).bbox)
        isneg(i) = 1;
    end
end
neginds = find(isneg);
nNeg = length(neginds);
negPerIm = round(NnegativeSamplesPerPositive / nNeg * nPos);
disp(['neg per image: ' num2str(negPerIm)]);


%totalNeg = negPerIm*floor(nNeg/2);    
totalNeg = negPerIm*nNeg;

if LOAD 
    load '../objdet/tmp/tmpdata.mat';
    count = min(find(sum(features, 2)==0))
    firstni = (count - nPos - 1) / negPerIm + 1
else


classes = zeros(nPos + totalNeg, 1);

features = zeros(nPos + totalNeg, Nfeatures + NcontextFeatures);

for p = 1:nPos
  
    %% use an image only if we have normalized version of it in 'data'

    disp(['pos: ' num2str(p)]);

    img = im2double(imread([imdir '/' imdata(p).nameImg]));
    load([imdir '/' strtok(imdata(p).nameImg, '.') '.c.mat']);
    cimages = cimages{1};
    
    % add padding if necessary    
%     if size(img, 1) < 20
%         img2 = 0.5*ones(20, size(img, 2));
%         img2(1:size(img, 1), 1:size(img, 2)) = img;
%         img = img2;
%     end
%     if size(img, 2) < 20
%         img2 = 0.5*ones(size(img, 1), 20);
%         img2(1:size(img, 1), 1:size(img, 2)) = img;
%         img = img2;
%     end    
    
    tic

    imfeatures = single(zeros([size(img,1) size(img,2) Nfeatures+NcontextFeatures]));

    for f = 1:Nfeatures        
        wavelet = dictionary(f).wavelet;
        exp_wavelet = dictionary(f).exp_wavelet;
        fragment = dictionary(f).fragment;
        gaussianX = dictionary(f).gaussianX;
        gaussianY = dictionary(f).gaussianY;
        exponent  = dictionary(f).exponent; % maybe another way of selecting the thresholds ...

        imfeatures(:,:,f) = single(convCrossConv(img, wavelet, ...
                           fragment, ...
                           gaussianY, gaussianX, exp_wavelet, exponent));  
                                                                             
    end
    imfeatures(:, :, Nfeatures+1:Nfeatures+NcontextFeatures) = ...
        detContextFeatures(cimages, detParams.objectSize);

    toc
    %subplot(131)
    %imagesc(img); title(ndxImage); colormap(gray(256)); axis('equal'); drawnow;
    %subplot(132)
    %imagesc(features(:,:,1));  colormap(gray(256)); axis('equal'); drawnow;
    %subplot(133)
    %imagesc(mean(features,3));  colormap(gray(256)); axis('equal'); drawnow;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    [nys, nxs, cols] = size(img);
    %imgBox = zeros(size(img));
    ndx = 1;    

    % Sampling the features output
    box = fix(imdata(p).box);
    x = fix(imdata(p).x); 
    y = fix(imdata(p).y);

    % Coordinates of the bounding box of the object:
    ym = max(1, box(3)); 
    yM = min([nys box(4)]);
    xm = max(1, box(1)); 
    xM = min([nxs box(2)]);
    %imgBox(ym:yM, xm:xM)=1;

    % Positive samples
    %dx = fix(3*rand)-1; 
    %dy = fix(3*rand)-1;    
    dx = 0;
    dy = 0;
    features(p, :) = imfeatures(min(max(y+dy,1),nys), min(max(x+dx,1),nxs), :);
    classes(p) =  1;
end

count = p;
npix = [];
clear imfeatures;

firstni = 1;

end

for ni = firstni:nNeg
    
    disp(['neg: ' num2str(ni)]);

    img = im2double(imread([imdir '/' gtruth(neginds(ni)).imname]));   

    if (size(img, 3)==3)
    
        img = rgb2gray(img);
        [tok, rem] = strtok(gtruth(neginds(ni)).imname, '/');
        fn = [imdir '/' tok '/context' strtok(rem, '.') '.c.mat'];    
        load(fn);
        cimages{1} = double(cimages{1});
        if max(size(img)) > maxImageSize
            scale = maxImageSize/max(size(img));
            img = imresize(img, scale, 'bilinear');
            tmp = zeros(size(img, 1), size(img, 2), size(cimages{1}, 3));
            for n = 1:size(cimages{1}, 3)
                tmp(:, :, n) = imresize(cimages{1}(:, :, n), scale, 'bilinear');
            end
            cimages{1} = tmp;
        end        

        imsize = size(img);
        for s = 1:100
            npix(s) = prod(imsize);
            imsize = imsize*scaleSampling;
            if min(imsize)<32
                break;
            end                
        end

        sc = 0;
        rnum = rand(1)*sum(npix);
        for rs = 1:length(npix)
            sc = sc+npix(rs);
            if rnum < sc
                break;
            end
        end    
        scale = scaleSampling^(rs-1); 
        disp(['scale: ' num2str(scale)]);
        if scale~=1
            img = imresize(img, scale, 'bilinear');
            cimages = imresize(cimages{1}, scale, 'bilinear');
        else
            cimages = cimages{1};
        end
        disp(num2str(size(img)));

        tic

        imfeatures = zeros([size(img,1) size(img,2) Nfeatures+NcontextFeatures], 'single');

        for f = 1:Nfeatures        
            wavelet = dictionary(f).wavelet;
            exp_wavelet = dictionary(f).exp_wavelet;
            fragment = dictionary(f).fragment;
            gaussianX = dictionary(f).gaussianX;
            gaussianY = dictionary(f).gaussianY;
            exponent  = dictionary(f).exponent; % maybe another way of selecting the thresholds ...

            imfeatures(:,:,f) = single(convCrossConv(img, wavelet, fragment, ...
                               gaussianY, gaussianX, exp_wavelet, exponent));  
        end    

        %size(contextFeatures(cimages, detParams.objectSize))
        imfeatures(:, :, Nfeatures+(1:NcontextFeatures)) = ...
            detContextFeatures(cimages, detParams.objectSize);


        toc

        for n = 1:negPerIm

            count = count + 1;

            posy = ceil(rand(1)*size(imfeatures, 1));
            posx = ceil(rand(1)*size(imfeatures, 2));

            features(count, :) = imfeatures(posy, posx, :);
            classes(count) = 0;
        end

        save '../objdet/tmp/tmpdata.mat' features classes

    end
        
end 

features = features(1:count, :);
classes = classes(1:count);

