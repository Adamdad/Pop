function out = convCrossConv(img, h, f, gy, gx, exp_h, exp_f) 
%
%  z =  ((img * h)^exp_h ** f)^exp_f * g

down_Sampling = 1;

% First, convolution: feature extraction
if max(size(h))>1
    out = conv2(img, h, 'same');
    out = abs(out).^exp_h;
else
    out = img;
end


if down_Sampling == 1
    out = downOctave(out);
    f = downOctave(f);
end


% Second, normalized correlation: template matching in feature representation
if max(size(f))>1
    [n, m] = size(f);
    out = normxcorr2(f, out); % XXX was normxcorr2(f, out)
    out = out(fix(n/2)+1:end-ceil(n/2)+1, fix(m/2)+1:end-ceil(m/2)+1);
    %out2 = normxcorr2_mex(f, out);
    %disp(num2str([0 size(out)]))
    %disp(num2str([1 size(out1)]))
    %disp(num2str([2 size(out2)]))
    %out = out1;
    out = out.^exp_f;    
end

if down_Sampling == 1
    out = imresize(out, size(img));
end

% Third, convolution: spatial localization
out = conv2(100*gy(end:-1:1), gx(end:-1:1), real(out), 'same');

