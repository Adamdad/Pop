function dictionary = detCreateDictionary(imdir, detParams, data)
% Create dictionary of fragments
%
% Each fragment is a triplet [h=wavelet, f=template, g=localizer], so that by
% computing:
%
% z = convCrossConvSeparable(img, h, f, gy, gx, exp_h, exp_f);
%
% 'z' is an image of the same size than 'img' with a gaussian located in
% the center of the object
%
% INPUT:
% imdir - the location of the images        
% detparams.objectSize(nsizes, [w h]): the size of the object window
%           Ntemplates: number of templates per object (suggest 500)
%           minSizeFragment: (suggest 6)
%           maxSizeFragment: (suggest 20)
% data: the data from tod_createData
% OUTPUT:
% dictionary - stores information needed to generate features



% PARAMETERS OF NORMALIZED OBJECTS:
objectSize      = detParams.objectSize; 
Ntemplates      = detParams.Ntemplates; 
minSizeFragment = detParams.minSizeFragment; 
maxSizeFragment = detParams.maxSizeFragment; 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1) Dictionary of wavelets:
wavelets = dictionaryFilters;
Nwavelets = length(wavelets);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2) dictionary templates:

set(gcf,'doublebuffer','on')

% Initializing vfigurearious things:
ndx = 0;
j = randperm(length(data));
%    j = j(1:NimagesToUse);
axy = (1:objectSize(1)) - objectSize(1)/2 - 1;

dictionary = repmat(struct('wavelet', [], 'exp_wavelet', [], 'fragment', [], ...
    'gaussianX', [], 'gaussianY', [], 'fileImg', [], 'exponent', []), 500, 1);

% Loop for creating the dictionary:
figure
while ndx<Ntemplates
    
    disp(num2str(ndx))
    J = j(fix(rand*length(data))+1);
    nameImg = data(J).nameImg;
    box = fix(data(J).box);
    x = data(J).x;
    y = data(J).y;

    img = im2double(imread([imdir '/' nameImg]));        
    if size(img, 3)==3
        img = rgb2gray(img);
    end        
    imgBox = zeros(size(img));
    [nys, nxs] = size(img);

    % coordinates of the bounding box of the object:
    ym = max(1, box(3)); yM = min(box(4), nys);
    xm = max(1, box(1)); xM = min(box(2), nxs);

    %disp([J ym yM xm xM])
    
    imgBox(ym:yM, xm:xM)=2;

    [yS, xS] = find(imgBox==2); % pixels of object support

    ntrials = 0; done = 0;

    % First select wavelet from dictionary:
    k = 1+fix(Nwavelets*rand);
    wavelet = wavelets(k).filter;
    exp_wavelet = fix(2*rand)+1;
    if max(size(wavelet))>1
        img = conv2(img, wavelet, 'same');
        if exp_wavelet==1
            img = abs(img);
        else                
            img = img.^exp_wavelet;
        end
    end

    if numel(xS) == 0
        disp('warning: xS empty')
    else
    
    while done == 0 & ntrials < 10
        % Dictionary intraclass segments
        ntrials = ntrials + 1;
        sizeFragment = minSizeFragment + fix((maxSizeFragment - minSizeFragment)*rand);

        Ds = 2*fix(sizeFragment/2)+1;
        N = floor(rand(2,1)*numel(xS))+1;
        x1 = xS(N(1));
        y1 = yS(N(2));

        if y1-(Ds-1)/2+1 > 0 & y1+(Ds+1)/2 <= nys & x1-(Ds-1)/2+1>0 & x1+(Ds+1)/2<=nxs
            fragment = img(y1-(Ds-1)/2+1:y1+(Ds+1)/2, x1-(Ds-1)/2+1:x1+(Ds+1)/2); % fragment object patch 
            gaussianX = exp( -(axy - (x1-x)).^2 / 3^2 ); 
            gaussianY = exp( -(axy - (y1-y)).^2 / 3^2 ); 

            done = 1;
        end
        %disp(num2str(done))
        %drawnow
    end    
    %disp(num2str(ndx))
    
    end
    
    if done        
        ndx = ndx+1;
        % Store fragment:
        dictionary(ndx).wavelet     = wavelet;
        dictionary(ndx).exp_wavelet = exp_wavelet;
        dictionary(ndx).fragment  = fragment;
        dictionary(ndx).gaussianX = gaussianX;
        dictionary(ndx).gaussianY = gaussianY;
        dictionary(ndx).fileImg   = nameImg;
       % dictionary(ndx).threshold = -2; % maybe another way of selecting the thresholds ...
        dictionary(ndx).exponent  = 2*fix(2*rand)+1; % maybe another way of selecting the thresholds ...
        
        if 0
        subplot(20,25,ndx)
        %imagesc(fragment); colormap(gray(256)); axis('off');
        %axis('equal'); axis('tight'); drawnow
        imagesc(fragment); colormap(gray(256)); axis('off'); axis('tight'); 
        %imshow(imresize(fragment, 2))
        
        drawnow
        end
    end
end

dictionary = dictionary(1:ndx);




