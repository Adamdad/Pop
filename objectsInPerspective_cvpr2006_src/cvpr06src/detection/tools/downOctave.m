function img = downOctave(img);

img = conv2([1 2 1]/4, [1 2 1]/4, img, 'same');
img = img(1:2:end,1:2:end);

