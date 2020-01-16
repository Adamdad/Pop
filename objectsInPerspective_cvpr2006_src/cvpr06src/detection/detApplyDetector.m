function score = detApplyDetector(Img, dict, paramClassifier, scaleSampling, maxImageSize)
%
% RUNS DETECTOR ON IMAGE
%
% scaleSampling = scaling steps (recomended = 0.85) (default is no scaling)
%
% maxImageSize  = it only runs the detector in the scales in which the
%                 image size is smaller than 'maxImageSize'. 
%
% score         = cell array. Each cell corresponds to one scale. The scales above
%               'maxImageSize' are filled with '-10000' values.


NweakClassifiers = length(paramClassifier);

img = double(mean(Img,3));
[rows, cols] = size(img);

if nargin < 5 
    maxImageSize = max([rows, cols]);
end

if nargin < 4 
    scaleSampling = 1;
end

scale = 0;
while min(size(img))>32
    scale = scale + 1;
    %disp(scale)
    Score = 0;
    if max(size(img)) <= maxImageSize 
        for j = 1:NweakClassifiers
            f  = paramClassifier(j).featureNdx;
            feature = convCrossConv(img, dict(f).wavelet, dict(f).fragment, dict(f).gaussianY, dict(f).gaussianX, dict(f).exp_wavelet, dict(f).exponent);
            Score = Score + paramClassifier(j).a * (feature > paramClassifier(j).th) + paramClassifier(j).b;
        end
    else
        % Score = -10000 * ones(size(img));
        Score = []; % DWH
    end
    
    if scaleSampling < 1
        score{scale} = Score;
        img = imresize(img, scaleSampling, 'bilinear');
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
    image(Img); axis('equal'); axis('tight');
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

