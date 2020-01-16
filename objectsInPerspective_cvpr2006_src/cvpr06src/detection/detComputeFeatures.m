function [features, classes]  = tod_computeFeatures(imdir, detParams, ...
    imdata, dictionary, gtruth, tmpsavefile)
% 3) Computes the input-output convolutional features 
%
% It creates a file with the features precomputed for each image in the
% labeledDatabase folder
%
% Input:
% imdir - the location of the images        
% detparams.NnegativeSamplesPerPositive: (suggest 100)
% data: the data from tod_createData
% dictionary: the data from tod_createDictionary
% gtruth(nimages).imname: the name of the image
%                 bbox(nobjs, [y1 y2 x2 x1]): the object bounding boxes
%                 
% Output:
% features{nimages}(ndata, nfeatures): feature data
% classes{nimages}(ndata): class labels

LOAD = 0;


% PARAMETERS:
NnegativeSamplesPerPositive = detParams.NnegativeSamplesPerPositive; 
USE_COLOR = 0;

nColorF = 0;
if USE_COLOR
    nColorF = 6;
end


% Compute features:

Nimages = length(gtruth);
Nfeatures = length(dictionary);

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

% don't process more negative images than necessary
if negPerIm < 50
    negPerIm = 50;
    nNeg = round(NnegativeSamplesPerPositive*nPos / negPerIm)
    rn = randperm(length(neginds));    
    neginds = neginds(rn(1:nNeg));
end
totalNeg = negPerIm*nNeg;

count = 0;
features = zeros(nPos + totalNeg, Nfeatures + nColorF);
classes = zeros(nPos + totalNeg, 1);

if LOAD
    load(tmpsavefile); 
    firstni = ni;
    if count < nPos
        count = 0;
    end
end

for p = (count+1):nPos
  
    %% use an image only if we have normalized version of it in 'data'

    disp(['pos: ' num2str(p)]);

    img = im2double(imread([imdir '/' imdata(p).nameImg]));
    
    if ~USE_COLOR || size(img, 3)==3    
    
        grayim = img;
        if size(grayim, 3)==3
            grayim = rgb2gray(grayim);
        end
    
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

        imfeatures = single(zeros([size(img,1) size(img,2) Nfeatures+nColorF]));

        for f = 1:Nfeatures        
            wavelet = dictionary(f).wavelet;
            exp_wavelet = dictionary(f).exp_wavelet;
            fragment = dictionary(f).fragment;
            gaussianX = dictionary(f).gaussianX;
            gaussianY = dictionary(f).gaussianY;
            exponent  = dictionary(f).exponent; % maybe another way of selecting the thresholds ...

            imfeatures(:,:,f) = single(convCrossConv(grayim, wavelet, ...
                               fragment, ...
                               gaussianY, gaussianX, exp_wavelet, exponent));  
        end


        if USE_COLOR
            imfeatures(:, :, f+1:end) = detColorFeatures(img, detParams.objectSize);
        end
        toc
        %toc
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
        dx = fix(3*rand)-1; 
        dy = fix(3*rand)-1;
        
        count = count + 1;
        
        features(count, :) = imfeatures(y+dy, x+dx, :);
        classes(count) =  1;
    end
end

if count <= nPos
    ni = 0;
    save(tmpsavefile, 'ni', 'neginds', 'features', 'classes', 'count');
    firstni = 1;
end



for ni = firstni:nNeg
    
    disp(['neg: ' num2str(ni)]);

    if mod(ni, 10)==0
        save(tmpsavefile, 'ni', 'neginds', 'features', 'classes', 'count');
    end
    
    img = im2double(imread([imdir '/' gtruth(neginds(ni)).imname]));

    if ~USE_COLOR || size(img, 3)==3
    
        % pick a random scale according to distribution of object windows
        scale = detParams.scaleSampling;
        maxk = min( (log(50)-log([size(img, 1) size(img, 2)]))/log(scale));
        roulette = cumsum(scale.^[0:maxk]);
        k = min(find(rand(1)*roulette(end) < roulette)) - 1;

        img = imresize(img, scale^k, 'bilinear');    

        if mean([size(img, 1) size(img, 2)])>500
            img = imresize(img, 0.5, 'bilinear');
        end

        grayim = img;
        if size(img, 3)==3
            grayim = rgb2gray(grayim);
        end    

        tic

        imfeatures = zeros([size(img,1) size(img,2) Nfeatures+nColorF], 'single');

        for f = 1:Nfeatures        
            wavelet = dictionary(f).wavelet;
            exp_wavelet = dictionary(f).exp_wavelet;
            fragment = dictionary(f).fragment;
            gaussianX = dictionary(f).gaussianX;
            gaussianY = dictionary(f).gaussianY;
            exponent  = dictionary(f).exponent; % maybe another way of selecting the thresholds ...

            imfeatures(:,:,f) = single(convCrossConv(grayim, wavelet, fragment, ...
                               gaussianY, gaussianX, exp_wavelet, exponent));  
        end    

        if USE_COLOR
            imfeatures(:, :, f+1:end) = detColorFeatures(img, detParams.objectSize);
        end    

        toc

        for n = 1:negPerIm

            count = count + 1;

            %[yn, xn] = find(imgBox==0);
            %N = length(xn(:)); N = randperm(N);        

            rind = ceil(rand(1)*size(img, 1)*size(img, 2));
            posy = ceil(rand(1)*size(img, 1));
            posx = ceil(rand(1)*size(img, 2));
            features(count, :) = imfeatures(posy, posx, :);
            classes(count) = 0;
        end
    end
end 

features = features(1:count, :);
classes = classes(1:count);
