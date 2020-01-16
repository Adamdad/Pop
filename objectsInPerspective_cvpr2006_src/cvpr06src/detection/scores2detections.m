function det = scores2detections(scores, objSize, step, varargin)
% det = scores2detections(scores, objSize, step, options)
% Converts the detection scores cell array to a set of bounding boxes and
% confidences.
%
% Input: 
%   scores{nscales,nsize}(ny, nx): confidences of detections at each pixel
%           and scale
%   objSize(nsize, [w h]): the size of an object window at original scale
%   step: the scaling factor (scores{s}(i, -) is has a y-value (row) of
%           i/sparam^(s-1) in the image)
%   options: name-value pairs
%       'minscore': the smallest score to be included in det
%       'maxdets': the maximum number of detections 
% Output:
%   det(ndet, [conf x1 x2 y1 y2])
%

% read options
minscore = -Inf;
maxdets = Inf;
options = varargin;
for n = 1:2:numel(options)
    if strcmp(lower(options{n}), 'minscore')
        minscore = options{n+1};
    elseif strcmp(lower(options{n}), 'maxdets')
        maxdets = options{n+1};
    else
        error(['Invalid input argument: ' options{n}])
    end
end

nsize = size(objSize, 1);
det = [];
for s = 1:nsize
    det = [det ; ...
        scores2detectionsP(scores(:, s), objSize(s, :), ...
        step, minscore, maxdets)];    
end

% sort rows in descending order and ensure that no more than maxdets boxes
% are included
det = -sortrows(-det, 1);
if size(det, 1)>maxdets
    det = det(1:maxdets, :);
end


% end scores2detections
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function det = scores2detectionsP(scores, objSize, step, minscore, maxdets)
% this is a private function that operates as described above, handling a
% single aspect ratio

% get detections that are greater than minscore
ndet = 0;
nscales = length(scores);
for s = 1:nscales
    ind{s} = find(scores{s}>minscore);       
    [y{s}, x{s}] = find(scores{s}>minscore);
    ndet = ndet + length(x{s});    
end

det = zeros(ndet, 5);
ndet = 0;
for s = 1:nscales
    len = length(x{s});
    sc = step^(1-s);
    det((ndet+1):(ndet+len), 1:5) = ...
        [scores{s}(ind{s}) sc*x{s} sc*y{s} repmat(sc*objSize, len, 1)];
    ndet = ndet+len;
end

% change det to form ([conf x1 x2 y1 y2])
x = det(:, 2);
y = det(:, 3);
hbox = det(:, 4:5)*0.5;
det(:, 2:5) = [x-hbox(:,1)+0.5 x+hbox(:,1)-0.5 y-hbox(:,2)+0.5 y+hbox(:,2)-0.5];



