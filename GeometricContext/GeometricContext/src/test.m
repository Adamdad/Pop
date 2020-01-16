% [labels, conf_map, maps, pmaps] = APPtestImage(city10,[],vert_classifier,horz_classifier,segment_density);
% imdir = 'E:\Matlab2019b\POP\popTestImages\popTestImages';

% imfn = dir([imdir '/*.jpg']);
% imfn = {imfn(:).name};
imdir = 'F:\cocotrain2017\train2017\train2017';
imfn = [];
fileID = fopen('F:/cocotrain2017/filenamelist.txt','r');
file = fgetl(fileID);
while ischar(file)
    imfn = [imfn, {file}];
%     disp(file)
    file = fgetl(fileID);
end
varargin = 'F:\cocotrain2017\context_result';
APPtestDirectory_o(segment_density, vert_classifier, horz_classifier, imdir, imfn, varargin);

