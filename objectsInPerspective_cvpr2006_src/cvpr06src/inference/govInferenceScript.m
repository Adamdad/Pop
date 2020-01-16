% govInferenceScript

% Bayes Network:
%   total nodes: 2*n_o+1 where n_o is num objects
%   node 1: camera state - values specify pair camera height, horizon       
%   node 2 to n_o+2: object state - last value specifies background,
%       value j=1..n_o specifies object presence at location (j-1)
%   node n_o+3:2*n_o+2: local geometry state corresponding to nodes
%       2 to n_o+1 before - values 1,2,3,4 represent notObjSurf/notGndBel,
%       notObjSurf/gndBel, objSurf/notGndBel, objSurf/gndBel
%   CPT: P(parent value, node value)

DO_INIT = 1;
DO_OBJ_DISPLAY = 1;
DO_VIEW_DISPLAY = 0;
DO_GIST_HORIZON = 1;

objTypes = [1 2]; % 1 = car, 2 = ppl

if DO_INIT

    basedir = 'E:\Matlab2019b\POP\popTestImages';
    imdir = [basedir '/popTestImages'];
    contextdir = [basedir '/context_result'];
    objdir = [imdir '/objectDetections'];
    datadir = 'E:\Matlab2019b\POP\objectsInPerspective_cvpr2006_src\cvpr06src\data';
    datadir2 = 'E:\Matlab2019b\POP\objectsInPerspective_cvpr2006_src\cvpr06src\data';
    
    % filenames to process
    load('E:\Matlab2019b\POP\popTestImages\popTestset.mat');
    fn = ts.imnames(1:600);
    
    % probabilities for estimating horizon (and standard deviation)  
    % new way
    %loadData = load([datadir '/labelme_horizprob2.mat']);
    %priorV_tmp = loadData.priorV;
    %priorV_tmp.x = priorV_tmp.x;    

    % old way (used for CVPR 2006)
    pH.sigma = 0.14; % prior from validation
    priorV_tmp.x = [0.0:0.01:0.99]';
    priorV_tmp.f = normpdf(priorV_tmp.x, 0.5, pH.sigma);        
    
    % use estimated horizons
    if DO_GIST_HORIZON
        loadData = load('E:\Matlab2019b\POP\objectsInPerspective_cvpr2006_src\cvpr06src\data/cvprHorizonEstimates.mat');
        horiz_mu = loadData.h_mu;
        horiz_b  = loadData.h_b;
    end
    
    % camera height prior
    loadData = load([datadir, '/labelme_cameraHeightPrior2.mat']);  % cvpr estimates
    if DO_GIST_HORIZON
        loadData = load('E:\Matlab2019b\POP\objectsInPerspective_cvpr2006_src\cvpr06src\data/priorY_lm.mat'); % ijcv estimates
    end
    priorY = loadData.priorY;  priorY.x = priorY.x(:);  priorY.f = priorY.f(:);        
    
    % probability of geometry given object, non-object
    loadData = load([datadir '/pG_car_ppl.mat']);
    pG = loadData.pG;
    
    % object detection candidates    
%     disp('loading dalal candidates')
%     loadData = load([datadir2 '/labelme_candidates_dalal_4.mat']);
%     %disp('loading MTF candidates')
%     %loadData =load([datadir '/cvprlabelme/labelme_results_o.mat']);
%     candidates = loadData.candidates;
        
end

% initialize viewpoint priors
priorV.x = 1-priorV_tmp.x;% + (1-horiz(f));
priorV.f = priorV_tmp.f;
pYV = cameraInitYV(priorY, priorV); 

for f = 1:length(fn) 

    disp([num2str(f) ': ' fn{f}])   
    
    % use viewpoint estimates to give image-specific horizon likelihood
    if DO_GIST_HORIZON
        pV.x = priorV.x;
        pV.f = 0.5*exppdf(abs(1-pV.x - horiz_mu(f)), horiz_b(f)); % double exponential
        pV.f = pV.f / sum(pV.f);        
        pYV = cameraInitYV(priorY, pV); 
    end
    
    loadData = load([basedir '/context_result/' strtok(fn{f}, '.') '.c.mat']);
    cimages = loadData.cimages{1};
    try
         [newcandidates{f}, pHorizon(f), pHeight(f)] = ...
        govInference(candidates{f}, cimages, pG, pYV);
    catch
        disp('fail!!!!!')
        continue
    end
   
             
    %% DISPLAY 
    if DO_OBJ_DISPLAY
        im = imread([imdir '/' fn{f}]);

        ncand = numel(candidates{f});
        
        gcf = figure(1), hold off, imshow(im), hold on
        for n = 1:ncand
            if candidates{f}(n).p > 0.1
                [tmp, ind] = max(candidates{f}(n).conf);
                bbox = candidates{f}(n).bbox(ind, :);
                if candidates{f}(n).objType == 1
                    plot(bbox([1 1 2 2 1]), bbox([3 4 4 3 3]), 'color',	[0,1,0] ,'linewidth',2);
                else
                    plot(bbox([1 1 2 2 1]), bbox([3 4 4 3 3]), 'color',	[0.5,1,0],'linewidth',2);
                end
            end
        end
        saveas(gcf,['E:\Matlab2019b\POP\objectsInPerspective_cvpr2006_src\cvpr06src\inference\\result/original',num2str(f),'.png'])
        %[maxval, maxind] = max(pV.f);
        %inith = 1-pV.x(maxind);
        %figure(1), plot([1 size(im, 2)], [inith inith]*size(im, 1), '--g', 'LineWidth', 2); 
        
        figure(2), hold off, imshow(im), hold on
        for n = 1:ncand
            if newcandidates{f}(n).p > 0.1
                [tmp, ind] = max(newcandidates{f}(n).conf);
                bbox = newcandidates{f}(n).bbox(ind, :);
                if newcandidates{f}(n).objType == 1
                    plot(bbox([1 1 2 2 1]), bbox([3 4 4 3 3]), 'color',	[1,0.25,0] ,'linewidth',2);
                else
                    plot(bbox([1 1 2 2 1]), bbox([3 4 4 3 3]), 'color',	[1,0,0.25] ,'linewidth',2);      
                end
                
            end
        end 

        gcf = figure(2), plot([1 size(im, 2)], [pHorizon(f).ml pHorizon(f).ml]*size(im, 1), '--r', 'LineWidth', 2); 
        
        saveas(gcf,['E:\Matlab2019b\POP\objectsInPerspective_cvpr2006_src\cvpr06src\inference\\result/after',num2str(f),'.png'])
        drawnow;
    end

    if DO_VIEW_DISPLAY
        figure(3), hold off, plot(pHorizon(f).x, pHorizon(f).f, 'g')
        figure(3), hold on, plot(1-priorV.x, priorV.f/sum(priorV.f), '--g')
        figure(4), hold off, plot(pHeight(f).x, pHeight(f).f, 'r')
        figure(4), hold on, plot(priorY.x, priorY.f/sum(priorY.f), '--r')    
        figure(5), imshow(reshape(pYV.f, numel(pYV.y), numel(pYV.v))'/max(pYV.f), 'XData', pYV.y, 'YData', 1-pYV.v, 'InitialMagnification', 'fit'), axis square, axis on
        %figure(6), imshow(reshape(marginal(1).T, numel(pYV.y), numel(pYV.v))'/max(marginal(1).T), 'XData', pYV.y, 'YData', 1-pYV.v, 'InitialMagnification', 'fit'), axis square, axis on                    
        drawnow;        
    end    

end
