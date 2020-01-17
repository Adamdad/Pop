% [labels, conf_map, maps, pmaps] = APPtestImage(city10,[],vert_classifier,horz_classifier,segment_density);
% imdir = 'E:\Matlab2019b\POP\popTestImages\popTestImages';

% imfn = dir([imdir '/*.jpg']);
% imfn = {imfn(:).name};
imdir = '/data/COCO/train2017';
imfn = [];
fileID = fopen('~/pop_project/Pop/GeometricContext/filenamelist.txt','r');
file = fgetl(fileID);
while ischar(file)
    imfn = [imfn, {file}];
    file = fgetl(fileID);
end
varargin = '~/pop_project/Pop/GeometricContext/coco_georesult_dir/';
APPtestDirectory_o(segment_density, vert_classifier, horz_classifier, imdir, imfn, varargin);

