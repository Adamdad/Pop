function z = crossConvSeparable(img, h, gy, gx, threshold, exponent) 
%
%  z =  [img ** h > th] * g > 0
%  g = separable filter = gx * gy
%
%  if threshold = -1
%    z = [img ** h] * g
%
%  z = ([img ** h]^gamma) * g

drawnow

% Mono-scale feature:
[n, m] = size(h);
[ny, nx, nc] = size(img);
Nfeatures = length(threshold);

z = normxcorr2(h, img); % was normxcorr2(h, img);
z = z(fix(n/2)+1:end-ceil(n/2)+1, fix(m/2)+1:end-ceil(m/2)+1);

if threshold > -1
    z = (z > threshold);
end

z = conv2(gy(end:-1:1), gx(end:-1:1), z.^exponent, 'same');




