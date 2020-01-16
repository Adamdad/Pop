function [det3, fn3] = detCombineDetections(det1, fn1, det2, fn2)
% [det3, fn3] = detCombineDetections(det1, fn1, det2, fn2)
% Combines detections in det1 and det2 according to corresponding image
% names given in fn1 and fn2
%
%   det1, det2, det3: cell array of array (ndet, [conf x1 x2 y1 y2])
%   fn1, fn2, fn3: list of image names; fn3 = fn1

% check inputs
%if numel(fn1)<numel(fn2) || numel(det1)<numel(det2)
%    error('fn2 and det2 must apply to subset of images in fn1');    
if ~iscell(det1) || ~iscell(fn1) || ~iscell(det2) || ~iscell(fn2)
    error('All inputs must be of type cell');
end

det3 = det1;
fn3 = fn1;

% strip path if any ('../test/image.jpg'==>'image.jpg')
for f = 1:numel(fn1)
    toks = strtokAll(fn1{f}, '/');
    fn1(f) = toks(end);
end
for f = 1:numel(fn2)
    toks = strtokAll(fn2{f}, '/');
    fn2(f) = toks(end);               
end

% concatenate det2 with det1
for f = 1:numel(fn1)
    ind = find(strcmp(fn1{f}, fn2));
    det3{f} = [det1{f} ; det2{ind}];
    if isempty(ind)
        warning(['did not find ' fn1{f}])
    end
end

