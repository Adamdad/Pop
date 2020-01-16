function det = detTest(imdir, gtruth, detector)

scaleSampling = 0.89;
maxImageSize = 500;
minScore = -2.0;

ndet = 0;
conf = cell(length(gtruth), 1);
bbox = cell(length(gtruth), 1);


for f = 1:length(gtruth)

        
    im = imread([imdir '/' gtruth(f).imname]);
        
    imsize = size(im);
                
    disp(num2str(f))
        
    if size(im, 3)==3
        im  = rgb2gray(im);
    end    
        %load([imdir '/context/' strtok(gtruth(f).imname, '.') '.c.mat']);
        %cimages = double(cimages{1});        
         
    scores = detApplyDetector(im,detector.dictionary, ...
        detector.classifier, scaleSampling, maxImageSize);
        
    detections = scores2detections(scores, detector.objectSize, scaleSampling, minScore);
                  
    if ~isempty(detections)
        conf{f} = detections(:, 1);
        bbox{f} = detections(:, 2:5);
        ndet = ndet + length(conf{f});
    end

    figure(1), hold off, imshow(im), hold on;
    for j = 1:max(1, size(detections, 1))
        plot(detections(j, [1 1 2 2 1]+1), detections(j, [3 4 4 3 3]+1),'r','linewidth',3);
    end      
    drawnow;
    %disp('press a key')
    %pause
    %close all

                  
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