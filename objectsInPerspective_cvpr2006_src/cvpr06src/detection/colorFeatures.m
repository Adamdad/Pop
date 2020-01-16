function fimage = colorFeatures(im, objsize)
%fimage = contextFeatures(cims, objsize)
% im - RGB image
% objsize - the size of the sliding window
% fimage(imsize, nfeatures) - the color features
%
% For a window being evaluated, the features are the average class
% confidences at that window, the average class confidences for a box of
% half-height directly below that window, and the difference between the
% values from the two boxes for each possible class

imsize = [size(im, 1) size(im, 2)];


fimage = zeros([imsize 6]);
scolimage = zeros([imsize 3]);

winfil = ones(objsize(2), objsize(1)); % windowed area
winfil = winfil / sum(winfil);
swinfil = ones(objsize(2)*2, objsize(1)*2); % surrounding area
swinfil(round(objsize(2)*0.5):round(objsize(2)*1.5), ...
    round(objsize(1)*0.5):round(objsize(1)*1.5)) = 0;
swinfil = swinfil / sum(swinfil);

% compute box means
for b = 1:3
    % middle
    fimage(:, :, b) = filter2(winfil, im(:, :, b), 'same');            
    % surround
    scolimage(:, :, b) = filter2(swinfil, im(:, :, b), 'same');
end

fimage(:, :, 1:3) = RGB2Lab(fimage(:, :, 1:3));

fimage(:, :, 4:6) = (fimage(:, :, 1:3) - RGB2Lab(scolimage)).^2;

