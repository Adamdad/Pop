
DO_PARALLEL = 0;
scaleSampling = 0.89;
maxImageSize = 800;
minScore = 0.0;

outdir = 'E:\Matlab2019b\POP\popValImages\objectDetections';
imdir = 'E:\Matlab2019b\POP\popValImages\popValImages';
%outdir = '/IUS/vmr19/dhoiem/labelme/valset/images/objectDetections';
%imdir = '/IUS/vmr19/dhoiem/labelme/valset/images';

fn = dir([imdir '/*.jpg']);
fn = {fn(:).name};

for f = 1:length(fn)        
        
    bn = strtok(fn{f}, '.');
    savename = [outdir '/' bn '.objdet.' num2str(detector(1).objectType) '.mat'];    

    if ~DO_PARALLEL || ~exist(savename)
    
        passeddet = [];
        save(savename, 'passeddet');
        
        im = im2double(imread([imdir '/' fn{f}]));

        imsize = size(im);

        disp(num2str(f))
     

        for o = 1:length(detector)        
            
            scores = detApplyDetectorDT(im,detector(o).dictionary, ...
                    detector(o), scaleSampling, maxImageSize);


            for s = 1:length(scores)
                passeddet(o).ind{s} = find(scores{s}>=minScore);
                passeddet(o).scores{s} = scores{s}(passeddet(o).ind{s});
                passeddet(o).size(s, 1:2) = size(scores{s});
                passeddet(o).imname = fn{f};
                passeddet(o).objsize = detector(o).objectSize;
                passeddet(o).index = f;
            end

            detections = scores2detections(scores', detector(o).objectSize, scaleSampling, 'minscore', minScore);
            
            if 1
            colors = 'rgb';
            figure(o), hold off, imshow(im), hold on;
            for j = 1:min(1, size(detections, 1))
                disp(['confidence: ' num2str(detections(j, 1))])
                plot(detections(j, [1 1 2 2 1]+1), detections(j, [3 4 4 3 3]+1),colors(j),'linewidth',3);
            end      
            drawnow;        
            end
        end

        disp(['saving to ' savename]);
        save(savename, 'passeddet');
        
    end
              
end
