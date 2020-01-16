function det = detTestContext(imdir, gtruth, detector)

scaleSampling = 0.89;
maxImageSize = 500;
minScore = -7.5;

ndet = 0;
conf = cell(length(gtruth), 1);
bbox = cell(length(gtruth), 1);
for f = 1:length(gtruth)
    
    disp(num2str(f))
    
    im = imread([imdir '/' gtruth(f).imname]);
    if size(im, 3)==3
        im  = rgb2gray(im);
        
        load([imdir '/context/' strtok(gtruth(f).imname, '.') '.c.mat']);
        cimages = cimages{1};        
                
        scores = detApplyDetectorContext(im, cimages, detector.dictionary, ...
            detector.classifier, scaleSampling, maxImageSize, detector.objectSize);
                
        detections = scores2detections(scores, detector.objectSize, scaleSampling, minScore);
                       
        conf{f} = detections(:, 1);
        bbox{f} = detections(:, 2:5);
        ndet = ndet + length(conf{f});
        
        figure(1), imshow(im)
        hold on; plot(bbox{f}(1, [1 1 2 2 1]),bbox{f}(1, [3 4 4 3 3]),'r','linewidth',3); hold off
        
        %disp('press a key')
        pause(1)
        %close all
        
    end
           
end

det = repmat(struct('imgnum', 0, 'confidence', 0, 'bbox', [0 0 0 0]), ndet, 1);
ndet = 0;

for f = 1:length(conf)    
    for k = 1:length(conf{f})
        det(ndet+k).imgnum = f;
        det(ndet+k).confidence = conf{f}(k);
        det(ndet+k).bbox = bbox{f}(k, :);
    end
    ndet = ndet + length(conf{f});
end

%pr = VOCpr(PASopts, testsets, det, true);