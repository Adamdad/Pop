function [det, list] = dalalOutput2Detections(detfn, listfn, marginxy)
% det = dalalOutput2Detections(detfn, listfn, marginxy)
% 
% Reads the text file resulting from Dalal's object detector from detfn and
% the corresponding image names from listfn and returns a cell array of
% bounding boxes.
%
% Input:
%   detfn: filename of object detector output
%   listfn: filename of list of image names
%   marginxy: pixel margin between object and bounding box at norm. size
% Output: 
%   det{nimages}(nboxes, [conf x1 x2 y1 y2]): results
%   list{nimages}: image names

data = textread(detfn, '%f', 'delimiter', ' ');

data = reshape(data, 7, numel(data)/7)';

% data(i, :) = x1 y1 width height scale conf num

% find starting position for each image
ind = find(data(:, 5)==0);

nimages = numel(ind);
ind(end+1) = size(data, 1)+1; % to mark end of last record

list = textread(listfn, '%s', 'delimiter', ' ');

if numel(list)~=nimages
    error('Number of images in detections does not match list')
end

det = cell(nimages, 1);

for i = 1:nimages
    irange = ind(i)+1:ind(i+1)-1;
    det{i} = data(irange, [1:4 6]);
    det{i}(:, 1:2) = det{i}(:, 1:2) + ...
        repmat(marginxy, size(det{i}, 1), 1).*data(irange, [5 5]);
    det{i}(:, 3:4) = det{i}(:, 1:2) + det{i}(:, 3:4) - 1 ...
        - 2*repmat(marginxy, size(det{i}, 1), 1).*data(irange, [5 5]);
    det{i} = det{i}(:, [5 1 3 2 4]);
end
    
        