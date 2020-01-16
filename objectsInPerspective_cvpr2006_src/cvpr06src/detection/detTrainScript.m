% detTrainScript
% For training object detectors based on the MTF2003 code (but
% significantly altered by Hoiem for CVPR06 code).

LOAD = 0;
CLEAR = 0;

imdir = '/IUS/vmr7/dhoiem/context/objdet/data/images/cars';
datadir = '/IUS/vmr7/dhoiem/context/objdet/data/oldCVPR06data';

objname = 'car';
VOCname = 'VOCcars';


%load([datadir '/c2p.mat']) % confidences to probabilities

if LOAD
    try
        load([datadir '/' objname 'car_gtruth.mat']);
        load([datadir '/' objname 'car_imdata.mat']);
        load([datadir '/' objname 'car_dictionary.mat']);
        load([datadir '/' objname 'car_features.mat']);
        load([datadir '/' objname 'car_detector.mat']);
    catch
    end
end

if CLEAR
    clear imdata dictionary features detector
end

% For CARS:
detparams.objectSize = [42 16 ; 32 22];% [36 20]; % was [32 20]
detparams.imageMaxSize = [100 100];
detparams.Ntemplates = 500;
detparams.minSizeFragment = 6;
detparams.maxSizeFragment = 20;
detparams.NnegativeSamplesPerPositive = 100;
detparams.NnegSamples = 10000;
detparams.percTrainingImages = 1.0;
detparams.NweakClassifiers = 100;
detparams.scaleSampling = 0.89;
detparams.maxNegSize = 500;

if ~exist('gtruth') 
    disp('ground truth')
    % get image names and bounding boxes
    pascaldir = '/IUS/vmr7/dhoiem/pascal/VOCdevkit/PASCAL/VOCdata';
    %setsToUse = {'train', 'val', 'test1', 'test2'};
    setsToUse = {'train', 'val'};
    gtruth = detInitData(pascaldir, VOCname, setsToUse, ...
        imdir, 1, detparams.objectSize);
    save([datadir '/' objname '_gtruth.mat'], 'gtruth');
end

if ~exist('imdata')
    disp('object data')
    imdata = detCreateData(imdir, detparams, gtruth); 
    save([datadir '/' objname '_imdata.mat'], 'imdata');
end

if ~exist('dictionary')
    disp('dictionary')
    for o = 1:size(detparams.objectSize, 1)
        tmpparams = detparams;
        tmpparams.objectSize = detparams.objectSize(o, :);
        tmpdata = imdata(find([imdata(:).boxnum]==o));
        dictionary{o} = detCreateDictionary(imdir, tmpparams, imdata);
    end
    save([datadir '/' objname '_dictionary.mat'], 'dictionary');
end

if ~exist('features') 
    disp('features')    
    disp('only second object size')
    for o = 1:size(detparams.objectSize, 1)
        tmpparams = detparams;
        tmpparams.objectSize = detparams.objectSize(o, :);
        tmpdata = imdata(find([imdata(:).boxnum]==o));         
        [features{o}, labels{o}]  = detComputeFeatures(imdir, ...
            tmpparams, tmpdata, dictionary{o}, gtruth, [datadir '/car_tmpfeatures2.mat']);
        save([datadir '/' objname '_features.mat'], 'features', 'labels');
    end
end

if ~exist('detector') 
    disp('detector')
    disp('warning: only first object size')
    for o = 1:size(detparams.objectSize, 1)
        ntrees = 15;
        nnodes = 8;
        % 2600 is roughly the number of object candidates for a 500x500
        % image with a 24x24 image at 0.5 max overlap
        prior = exp(-3.5); 
        detector(o) = trainLogitBoostDt(features{o}, labels{o}*2-1, ...
            ntrees, nnodes, prior);                        
        detector(o).objectSize = detparams.objectSize(o, :);        
        detector(o).objectType = 1;  
        detector(o).dictionary = dictionary{o};            
    end
    save([datadir '/' objname '_detector.mat'], 'detector');
end

save([datadir '/' objname '_training_data.mat'], 'gtruth', 'imdata', 'dictionary', ...
    'features', 'labels', 'detector');
    



                     

