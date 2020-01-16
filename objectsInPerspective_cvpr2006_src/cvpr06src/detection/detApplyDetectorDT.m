function score = detApplyDetectorDT(im, dict, detector, scaleSampling, maxImageSize)
% score = detApplyDetectorDT(im, dict, detector, scaleSampling, maxImageSize)
%
% Runs detector on image
% Inputs:
%   im: the input image
%   dict: the dictionary of features that are used
%   detector: the full object detector
%   scaleSampling: scaling steps (recomended = 0.89)
%   maxImageSize: it only runs the detector in the scales in which the
%                 image size is smaller than 'maxImageSize'. 
% Output:
%   score: cell array. Each cell corresponds to one scale. The scales above
%         'maxImageSize' are set to be empty arrays

USE_COLOR = 1;

origim = im;

if size(im, 3) == 3
    grayim = rgb2gray(im);
else
    grayim = im;    
end
origgrayim = grayim;

nColorFeatures = 6;
if ~USE_COLOR
    nColorFeatures = 0;
end


[featureInds, nnormalFeatures, detector] = ...
   makeCompactDetector(detector, dict, nColorFeatures);
%nnormalFeatures = 500;
%length(featureInds)

[rows, cols] = size(im);

if nargin < 5 
    maxImageSize = max([rows, cols]);
end

if nargin < 4 
    scaleSampling = 1;
end

scale = 0;
while min(size(grayim))>32
    scale = scale + 1;
    %disp(scale)
    Score = 0;
    if max(size(grayim)) <= maxImageSize 
        imfeatures = zeros(size(grayim, 1)*size(grayim, 2), ...
            nnormalFeatures+nColorFeatures, 'single');    
        imfsize = size(imfeatures);
        
        if USE_COLOR
            imfeatures(:, nnormalFeatures+1:nnormalFeatures+6) = ...
                reshape(detColorFeatures(im, detector.objectSize), imfsize(1), 6);
        end            
        
        for j = 1:length(featureInds)
            f = featureInds(j);
            if f <= length(dict)
                imfeatures(:, j) = reshape(convCrossConv(grayim, dict(f).wavelet, ...
                    dict(f).fragment, dict(f).gaussianY, dict(f).gaussianX, ...
                    dict(f).exp_wavelet, dict(f).exponent), imfsize(1), 1);
            end            
        end

        Score = testBoostedDtMc(detector, imfeatures);
        Score = reshape(Score, size(grayim));
    else
        % Score = -10000 * ones(size(img));
        Score = []; % DWH
    end
    
    if scaleSampling < 1
        score{scale} = Score;
        if mod(scale, 4)==0            
            grayim = imresize(origgrayim, scaleSampling^scale, 'bilinear');
            if USE_COLOR
                im = imresize(origim, scaleSampling^scale, 'bilinear');            
            end            
        else
%             grayim = imresize(grayim, scaleSampling, 'bilinear',5);
            grayim = imresize(grayim, scaleSampling, 'bilinear');
            if USE_COLOR
%                 im = imresize(im, scaleSampling, 'bilinear',5);
                im = imresize(im, scaleSampling, 'bilinear');
            end
        end
    else
        score = Score;
        break
    end    
end

%%%%%%%%%%%%%%%%%%
% VISUALIZATION.
% Only runs if there are no output arguments when calling the function.
%%%%%%%%%%%%%%%%%%

if nargout == 0
    if scaleSampling==1; score = {score}; end
    Nscales = length(score);
    
    % Find dynamic range of score:
    n=0;
    for i = 1:Nscales
        m = min(min(score{i}));
        if m>-1000
            n=n+1;
            Min(n) = m;
            Max(n) = max(max(score{i}));
        end
    end
    Min = min(Min);
    Max = max(Max);
    
    multiscaleScore = -1000;
    clear x y m
    figure('name', 'Raw output of the classifier for each scale')
    nx = fix(sqrt(Nscales)); ny = ceil(Nscales/nx);
    for i = 1:Nscales
        m(i) = max(max(score{i}));
        [yy, xx] = find(score{i} == m(i));
        x(i) = xx(1); y(i) = yy(1);

        subplot(ny,nx,i)
        S = score{i};
        S = uint8((S-Min)/(Max-Min)*255);
        image(S); 
        colormap(hot(256))
        colorbar
        axis('equal'); axis('tight');drawnow
        
        multiscaleScore = max(multiscaleScore, imresize(score{i}, [rows, cols]));
    end
    [M, bestScale] = max(m);

    scaling = (1/scaleSampling)^(bestScale-1);
    x = x(bestScale)*scaling;
    y = y(bestScale)*scaling;
    L = 16*scaling;
    disp(num2str([x y]))
    figure ('name', 'Output of the detector')
    subplot(121)
    image(grayim); axis('equal'); axis('tight');
    hold on; plot([x-L x-L x+L x+L x-L],[y-L y+L y+L y-L y-L],'r','linewidth',3); hold off
    title(sprintf('Run on %d scales. The maximum is at scale %d.', Nscales, bestScale))
    xlabel('Only the maximum output is shown')
    colormap(gray(256))
    subplot(122)
    S = multiscaleScore;
    S = uint8((S-Min)/(Max-Min)*255);
    image(S);
    title('Max across scales of raw classifier output')
    colormap(hot(256))
    colorbar
    axis('equal'); axis('tight');
    drawnow
end



%%%%%%%%%%%%%%%%%

function [featureInds, nnormalFeatures, detector] = ...
    makeCompactDetector(detector, dict, ncolor);
% change variable names to be consecutive

usedFeatures = detGetUsedFeatures(detector);
featureInds = find(usedFeatures>0);

nnormalFeatures = max(find(featureInds<=length(dict)));
for w = 1:numel(detector.wcs)
    dt = detector.wcs(w).dt;
    for f = 1:numel(dt.var)
        if dt.var(f) > 0 && dt.var(f) <= length(dict)
            detector.wcs(w).dt.var(f) = find(featureInds==dt.var(f));
        elseif dt.var(f) > 0
            detector.wcs(w).dt.var(f) = dt.var(f) - length(dict) + nnormalFeatures;
        end
    end    
    
    detector.wcs(w).dt.npred = nnormalFeatures + ncolor;
end

