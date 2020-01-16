% detReadCandidatesScript: reads candidates from MTF and dalal classifiers

MAX_DET = 1000; % max detections per type per image
MIN_PROB = 0.0005; % min marginal probability to be included
OV_THRESH = 0.33;  % overlap threshold for merging candidates

datadir = 'E:\Matlab2019b\POP\objectsInPerspective_cvpr2006_src\cvpr06src\data';

% get image names to read
%load('/IUS/vmr20/dhoiem/data/testset2.mat');
%fn = ts.imnames(ts.good_inds);
load('E:\Matlab2019b\POP\popValImages\popValset.mat');
fn = vs.imnames(1:60);
sources = {};

% set object types to include: 1 = car, 2 = ppl
objtypes = [1,2];

% set sigmoid parameters to convert score to prob: 
% p = 1 / (1+exp(A*score+B) ;  param(type, :) = [A B]
%mtfParam(1, :) = [-0.64 9.14]; % mtf cars when using entire results
%mtfParam(2, :) = [-0.84 9.67]; % mtf people when using entire results

% set source information

% MurphyTorralbaFreeman
objdetdir = 'E:\Matlab2019b\POP\popValImages\objectDetections';
bboxrange(1, :) = [0 Inf 0 32]; % [minw maxw minh maxh] car size range
bboxrange(2, :) = [0 Inf 0 96]; % [minw maxw minh maxh] ppl size range
%mtfParam(1, :) = [-0.70 11.76]; % mtf cars when using only small boxes from 24
mtfParam(1, :) = [-0.63 10.25]; % mtf cars when using only small boxes from 32
mtfParam(2, :) = [-0.62 8.67]; % mtf people when using only small boxes

% not using MTF
sources{1} = {'mtf', {objdetdir, objdetdir}, mtfParam, bboxrange};

% DalalTriggs (cars, big people)
% outfncar = [datadir 'dalalCarRaw.txt'];
% outfnppl = [datadir 'dalalPedBigRaw.txt'];
% listfn = [datadir 'imagelist.txt'];
% bboxrange(1, :) = [0 Inf 0 Inf]; % [minw maxw minh maxh] car size range
% bboxrange(2, :) = [0 Inf 96 Inf]; % [minw maxw minh maxh] ppl size range
% dalalParam(1, :) = [-4.23 2.31]; % dalal cars for 1 of 1
% dalalParam(2, :) = [-3.00 2.92]; % dalal ppl
%dalalParam(1, :) = [-2.89 2.31]; % dalal cars for 1 of 3
% margins = [16 16];
% sources{1} = {'dalal', {outfncar, outfnppl}, {listfn, listfn}, dalalParam, bboxrange, margins};

% DalalTriggs (small people)
% outfnppl = [datadir 'dalalPedSmRaw.txt'];
% bboxrange(1, :) = [0 Inf 0 Inf]; % [minw maxw minh maxh] car size range
% bboxrange(2, :) = [0 Inf 0 96]; % [minw maxw minh maxh] ppl size range
% %dalalParam(1, :) = [-4.23 2.31]; % dalal cars for 1 of 1
% dalalParam(2, :) = [-2.77 3.79]; % dalal ppl
% %dalalParam(1, :) = [-2.89 2.31]; % dalal cars for 1 of 3
% margins = [8 8];
% sources{2} = {'dalal', {[], outfnppl}, {[], listfn}, dalalParam, bboxrange, margins};

% outfncar = '/IUS/vmr20/dhoiem/datasets/dalal/cars/OLT/results/out2.txt';
% dalalParam(1, :) = [-4.62 -0.50]; % dalal cars for 2 of 3
% sources{2} = {'dalal', {outfncar, []}, {listfn, []}, dalalParam, bboxrange};
% 
% outfncar = '/IUS/vmr20/dhoiem/datasets/dalal/cars/OLT/results/out3.txt';
% dalalParam(1, :) = [-2.62 0.08]; % dalal cars for 3 of 3
% sources{3} = {'dalal', {outfncar, []}, {listfn, []}, dalalParam, bboxrange};


candidates = detReadAllCandidates(fn, objtypes, sources, OV_THRESH, MIN_PROB, MAX_DET);