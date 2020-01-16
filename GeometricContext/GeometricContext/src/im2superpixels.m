function imsegs = im2superpixels(im)

prefix = num2str(floor(rand(1)*10000000));
% fn1 = ['./tmpim' prefix '.ppm'];
fn2 = ['./tmpimsp',prefix,'.ppm'];
% segcmd = '/home/dhoiem/cmu/tools/superpixelsFH/segment 0.8 100 100';
% 
% imwrite(im, fn1);
% system([segcmd ' ' fn1 ' ' fn2]);
[L,N] = superpixels(im,100,'NumIterations', 100);
L=double(L)/max(N);
imwrite(L,fn2);
imsegs = processSuperpixelImage(fn2);

delete(fn2);