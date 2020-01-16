function fimage = detContextFeatures(cims, objsize)
%fimage = detContextFeatures(cims, objsize)
%
% Input:
% cims(imheight, imwidth, nclasses) - confidence images, one for each class
% objsize - the size of the sliding window
% c2p - structure for converting confidences to probabilities (deprecated)
%
% Output:
% fimage(imsize, nfeatures) - the context features
%
% For a window being evaluated, the features are the average class
% confidences at that window, the average class confidences for a box of
% half-height directly below that window, and the difference between the
% values from the two boxes for each possible class

imsize = [size(cims, 1) size(cims, 2)];
nclasses = size(cims, 3);

fimage = zeros([imsize nclasses*3+6]);

winfil = ones(objsize(2), objsize(1)); % object window
winfil = winfil / sum(winfil);
bwinfil = ones(objsize(2)*2, objsize(1)); % below object window
bwinfil(1:round(objsize(2)*1.5), :) = 0;
bwinfil = bwinfil / sum(bwinfil);
swinfil = ones(objsize(2)*2, objsize(1)*2); % around object window
swinfil(round((objsize(2)/2+1):2*objsize(2)), ...
    round((objsize(1)/2+1):3*objsize(1)/2)) = 0;    
swinfil = swinfil / sum(swinfil);

% compute box sums
for n = 1:nclasses
    % middle
    fimage(:, :, n) = filter2(winfil, cims(:, :, n), 'same');            
    % bottom
    fimage(:, :, n+nclasses) = filter2(bwinfil, cims(:, :, n), 'same');
    % top
    fimage(:, :, n+2*nclasses) = filter2(swinfil, cims(:, :, n), 'same');
end

% set constants where entire window is outside of image (rare to never)
[y, x] = find(fimage(:, :, 1)==0);
for i = 1:length(x)
    fimage(y(i), x(i), 1:nclasses) = 1;
end
[y, x] = find(fimage(:, :, nclasses+1)==0);
for i = 1:length(x)
    fimage(y(i), x(i), nclasses+1:2*nclasses) = 1;
end



% normalize and convert to probabilities
for k = [0:2]*nclasses

    % set constants where entire window is outside of image (rare to never)
    [y, x] = find(fimage(:, :, k+1)==0);
    for i = 1:length(x)
        fimage(y(i), x(i), k+(1:nclasses)) = 1;
    end    
    sum_1_3 = sum(fimage(:, :, k+(1:3)), 3);
    sum_4_8 = sum(fimage(:, :, k+(4:8)), 3);
    for n = 1:3
        fimage(:, :, k+n) = fimage(:, :, k+n) ./ sum_1_3;
    end
    for n = 4:8
        % multiply by likelihoood of vertical
        fimage(:, :, k+n) = fimage(:, :, k+2) .* fimage(:, :, k+n) ./ sum_4_8;
    end
    
    % convert to probs    
%     for n = 1:8
%         sp = c2p.obj(n).sigmoidp;
%         p = fimage(:, :, k+n);
%         fimage(:, :, k+n) = 1./(1+exp(log(p ./ (1-p))*sp(1)+sp(2)));     
%     end    
end

% compute:
% 1) P(obj=solid, bel=ground, sur=non-solid)
% 2) P(obj=solid, bel=ground)
% 3) P(obj=vert, bel=ground)
% 4) P(obj=solid, sur=non-solid)
% 5) P(obj=vert, sur=non-vert)
% 6) P(obj=vert, bel=ground, sur=non-vert)
nc = nclasses;
fimage(:, :, 3*nc+1) = fimage(:, :, 8).*fimage(:, :, nc+1).*(1-fimage(:, :, 2*nc+8));
fimage(:, :, 3*nc+2) = fimage(:, :, 8).*fimage(:, :, nc+1);
fimage(:, :, 3*nc+3) = fimage(:, :, 2).*fimage(:, :, nc+1);
fimage(:, :, 3*nc+4) = fimage(:, :, 8).*(1-fimage(:, :, 2*nc+8));
fimage(:, :, 3*nc+5) = fimage(:, :, 2).*(1-fimage(:, :, 2*nc+2));
fimage(:, :, 3*nc+6) = fimage(:, :, 2).*fimage(:, :, nc+1).*(1-fimage(:, :, 2*nc+2));
