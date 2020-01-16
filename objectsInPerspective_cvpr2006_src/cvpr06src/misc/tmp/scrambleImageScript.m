% scrambleImageScript

[imh imw imb] = size(im);

pw = 24;  ph = 60;
sp = 5;

npw = floor(imw / pw);
nph = floor(imh / ph);

im2 = zeros(nph*ph + (nph-1)*sp, npw*pw + (npw-1)*sp, imb);

rp = randperm(npw*nph);
px = floor((rp-1)/nph);
py = mod(rp-1, nph);

for k = 1:numel(rp)
    imx = floor((k-1)/nph)*pw;
    imy = mod(k-1, nph)*ph;
    
    patch = im(imy+1:imy+ph, imx+1:imx+pw, :);
    
    im2x = px(k)*(pw+sp);
    im2y = py(k)*(ph+sp);
    
    im2(im2y+1:im2y+ph, im2x+1:im2x+pw, :) = patch;
    
end

