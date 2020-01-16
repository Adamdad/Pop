% testCameraObjectBP_script

% Bayes Network:
%   total nodes: 2*n_o+1 where n_o is num objects
%   node 1: camera state - values specify pair camera height, horizon       
%   node 2 to n_o+2: object state - last value specifies background,
%       value j=1..n_o specifies object presence at location (j-1)
%   node n_o+3:2*n_o+2: local geometry state corresponding to nodes
%       2 to n_o+1 before - values 1,2,3,4 represent notObjSurf/notGndBel,
%       notObjSurf/gndBel, objSurf/notGndBel, objSurf/gndBel
%   CPT: P(parent value, node value)

INIT = 1;
%LOAD_CANDIDATES = 1;

outdir = '/IUS/vmr7/dhoiem/context/results/CVPR06/qualitative12';

%disp('ONLY PEOPLE')
objTypes = [1 2];

if INIT

    basedir = '/IUS/vmr20/dhoiem/datasets/testset';
    imdir = [basedir '/allimages'];
    contextdir = [basedir '/context'];
    objdir = [imdir '/objectDetections'];
    datadir = '/IUS/vmr7/dhoiem/context/objdet/data';

    load('/IUS/vmr20/dhoiem/data/testset2.mat');
    fn = ts.imnames(ts.good_inds);
    
    %imfiles = dir([imdir '/*.jpg']);
    %fn = {imfiles(:).name};

    % probabilities for estimating horizon (and standard deviation)
    loadData = load([datadir '/labelme_horizprob.mat']);
    pH = loadData.pH;
    priorV_tmp.x = [-0.5:0.01:0.5]';
    priorV_tmp.f = normpdf(priorV_tmp.x, 0, pH.sigma);
    % horizon estimates
    loadData = load([datadir '/labelme_horiz2.mat']);
    horiz = loadData.horiz;
    % camera height prior
    loadData = load([datadir '/labelme_cameraHeightPrior.mat']);
    priorY = loadData.priorY;
    % load detector settings
    loadData = load([datadir '/car_detector.mat']);
    detector{loadData.detector.objectType} = loadData.detector;
    loadData = load([datadir '/ppl_detector.mat']);
    detector{loadData.detector.objectType} = loadData.detector;
    % probability of geometry given object, non-object
    loadData = load([datadir '/pG_car_ppl.mat']);
    pG = loadData.pG;

    loadCand = load([datadir '/labelme_results_o.mat']);
end



%objprior = [1/100000 1/100000];
%objpriorLR = log(objprior ./ (1-objprior));
%objpriorLR = [-13.5 -11.4];
objpriorLR = [-11.2 -10.4];
cfactor = [0.82 0.91];
%objpriorLR = [-11 -11];
objprior = 1./(1+exp(-objpriorLR));
minconf = [0.001 0.01];

%rind = randperm(600);
%disp('warning: starting at 203')
for f = 38  %length(fn) %ts.good_inds
   % [38 144 243 262]
    disp([num2str(f) ': ' fn{f}])

%    disp('warning: using rigged horizon and sigma')
    %pH.sigma = 0.14; % prior from validation
    horiz(f) = 0.50;
        
    CPD = {};

    % get camera probabilities
    priorV.x = priorV_tmp.x + (1-horiz(f));
    priorV.f = priorV_tmp.f;
    pYV = cameraInitYV(priorY, priorV);    
    % remove very unlikely combinations of camera height and horizon
    [sf, sind] = sort(pYV.f, 'ascend');
    [tmp, ind] = min(abs(cumsum(sf)-0.015));
    pyvind = sind(ind:end);
    pYV.f = pYV.f(pyvind);
    pYV.f = pYV.f / sum(pYV.f);
    pYV.ymat = pYV.ymat(pyvind);
    pYV.vmat = pYV.vmat(pyvind);
    bn = strtok(fn{f}, '.');
    candidates = {};
    
    iminfo = imfinfo([imdir '/' fn{f}]);
    imsize = [iminfo.Height iminfo.Width];
    
    % get object candidates        
    if ~exist('loadCand')                %LOAD_CANDIDATES
        for t = 1:length(objTypes)

            loadData = load([objdir '/' bn '.objdet.' num2str(objTypes(t)) '.mat']);    
            objdet = loadData.passeddet;
            scores = {};  
            for sz = 1:length(objdet)
                for sc = 1:size(objdet(sz).size, 1)
                    scores{sc, sz} = repmat(0, objdet(sz).size(sc, :));
                    scores{sc, sz}(objdet(sz).ind{sc}) = ...
                        1./(1+exp(-(cfactor(objTypes(t))*objdet(sz).scores{sc}+objpriorLR(objTypes(t)))));
                end              
            end
            sparam = 0.89;
            objsize = reshape([objdet(:).objsize], 2, numel(objdet))';        
            candidates{t}  = scores2candidates2(scores, sparam, objsize, 0.5, minconf(t));
            for k = 1:numel(candidates{t})
                candidates{t}(k).objType = objTypes(t);
                candidates{t}(k).objPrior = objprior(t);
            end

        end
        candidates = [candidates{:}];
    else
        %loadData = load(['../objdet/data/candidates/candidates.' num2str(f) '.mat']);
        %candidates = loadData.candidates;
        candidates = [];
        for t = objTypes
            ind = find([loadCand.candidates{f}(:).objType]==t);
            %if numel(ind > 10)
            %    ind = ind(1:10);
            %    disp('reducing ind to 10')
            %end
            candidates = [candidates loadCand.candidates{f}(ind)];
        end        
    end
    ncand = numel(candidates);
    
    %candidates = candidates(1);
%     candidates = [];
%     candidates.p = 0.99;
%     candidates.bbox = [200 250 250 300];
%     candidates.conf = candidates.p;
%     candidates.objType = 1;
    
    % count number of nodes and get node sizes
    nnodes = 1 + 2*ncand;
    disp(['nnodes: ' num2str(nnodes)])
    node_sizes = zeros(1, nnodes);
    node_sizes(1) = numel(pYV.f);
    for k = 1:ncand
        node_sizes(k+1) = numel(candidates(k).conf)+1; % object node
        node_sizes(k+1+ncand) = 3; % geom node
    end
    
    % create DAG and bayes net
    dag = zeros(nnodes,nnodes);
    dag(1, 2:(ncand+1)) = 1; % camera to object links
    for k = 1:ncand % object to geom links
        dag(k+1, k+1+ncand) = 1;
    end
    bnet = mk_bnet(dag, node_sizes, 'discrete', [1:nnodes]);
    
    % create conditional probability tables
    bnet.CPD = cell(1, nnodes);
    
    % camera variables
    bnet.CPD{1} = tabular_CPD(bnet, 1, 'CPT', pYV.f);
    
    % objects
    for k = 1:ncand
        cpt = probObjectGivenCamera(candidates(k), pYV, imsize);
        bnet.CPD{k+1} = tabular_CPD(bnet, k+1, 'CPT', cpt);
    end
        
    % geometry
    loadData = load([basedir '/context/' strtok(fn{f}, '.') '.c.mat']);
    for k = 1:ncand
        cpt = probGeomGivenObject(candidates(k), pG, loadData.cimages{1});
        bnet.CPD{ncand+k+1} = tabular_CPD(bnet, ncand+k+1, 'CPT', cpt{1});
    end
    
    % do marginalization
    evidence = cell(1, nnodes);
    engine = pearl_inf_engine(bnet, 'protocol', 'tree');
    %engine = jtree_inf_engine(bnet);
    engine = enter_evidence(engine, evidence, 'maximize', 0);
    clear marginal;
    for n = 1:nnodes
        marginal(n) = marginal_nodes(engine, n);
    end
    newcandidates{f} = candidates;
    for n = 2:(ncand+1)
        pobj = sum(marginal(n).T(1:end-1));
        newcandidates{f}(n-1).p = pobj;
        newcandidates{f}(n-1).conf = marginal(n).T(1:end-1);
    end
    %engine = jtree_inf_engine(bnet);

    %mpe = calc_mpe(engine{1}, []);

    

    
    %% DISPLAY 
    im = imread([imdir '/' fn{f}]);
    %pause(1)

    figure(1), hold off, imshow(im), hold on
    plot([1 size(im, 2)], [horiz(f) horiz(f)]*size(im, 1), 'g--', 'LineWidth', 3);
    for n = 1:ncand
        if candidates(n).p > 0.3
            [tmp, ind] = max(candidates(n).conf);
            %disp(num2str(candidates(n).p))
            bbox = candidates(n).bbox(ind, :);
            plot(bbox([1 1 2 2 1]), bbox([3 4 4 3 3]), 'g' ,'linewidth',2);
        end
    end    

    figure(2), hold off, imshow(im), hold on;        
    [tmp, hind] = max(marginal(1).T);
    
    h2 = 1-pYV.vmat(hind);
    plot([1 size(im, 2)], [h2 h2]*size(im, 1), 'r', 'LineWidth', 3);    
    
    colors = 'bg';
    count = 0;
    for k = 2:ncand+1
        pobj = sum(marginal(k).T(1:end-1));
        if pobj > 0.3
            %disp(num2str(pobj))
            [tmp, ind] = max(marginal(k).T(1:end-1));            
            bbox = candidates(k-1).bbox(ind, :);
            count = count + 1;
            %c = colors(mod(count-1,6)+1);
            c = colors(candidates(k-1).objType);
            plot(bbox([1 1 2 2 1]), bbox([3 4 4 3 3]), c ,'linewidth',2);
            %pause(1)
        end
    end 
    
    
    % do 1-val so that 0 is the top of the image 
    pYV.vmat = 1-pYV.vmat;
    pYV.v = 1-pYV.v;
    [pHorizon(f), pHeight(f)] = cameraJointToMarginals(pYV, marginal(1).T);
    figure(2), plot([1 size(im, 2)], [pHorizon(f).exp pHorizon(f).exp]*size(im, 1), 'b', 'LineWidth', 2); 
    %figure(3), hold off, plot(pHorizon(f).x, pHorizon(f).f, 'g')
    %figure(3), hold on, plot(1-priorV.x, priorV.f/sum(priorV.f), '--g')
    %figure(4), hold off, plot(pHeight(f).x, pHeight(f).f, 'r')
    %figure(4), hold on, plot(priorY.x, priorY.f/sum(priorY.f), '--r')
    
    bn = strtok(fn{f}, '.');
    
    toutdir = outdir;
    try, mkdir(outdir, bn);
    catch, end;
    outdir = [outdir '/' bn];
    
    % WRITE ORIGINAL IMAGE
    imwrite(im, [outdir '/' fn{f}]);  
    
    % WRITE INITIAL VIEWPOINT PROB
    pYV.f2 = zeros(numel(pYV.y), numel(pYV.v));
    mf = max(pYV.f);
    pYV.f2(pyvind) = pYV.f / sum(pYV.f); % / max(pYV.f);    
    [y, v] = find(pYV.f2 > 1E-7);

    ymn = min(y); ymx = max(y); vmn = min(v);  vmx = max(v);
    pmn = min(pYV.f2(:)); pmx = max(pYV.f2(:));
    mYV2 = zeros(numel(pYV.y), numel(pYV.v));
    mYV2(pyvind) = marginal(1).T;% / max(marginal(1).T);
    [y, v] = find(mYV2 > max(marginal(1).T)/100);
    ymn2 = min(y); ymx2 = max(y); vmn2 = min(v);  vmx2 = max(v);
    pmn2 = min(mYV2(:)); pmx2 = max(mYV2(:));
    ymn = min(ymn, ymn2); ymx = max(ymx, ymx2); 
    vmn = min(vmn, vmn2); vmx = max(vmx, vmx2);
    pmn = min(pmn, pmn2); pmx = max(pmx, pmx2);    
        
    znorm = abs((pYV.y(2)-pYV.y(1))*(pYV.v(1)-pYV.v(2)));
    figure(3), surf(1-pYV.v(vmn:vmx),pYV.y(ymn:ymx), pYV.f2(ymn:ymx,vmn:vmx)/znorm), axis vis3d %axis square, axis on    
    set(gca, 'ZGrid', 'off'), grid off, shading interp, view(-60, 30)      
    set(gca,'ZTick',[pmn pmx])
%    set(gca,'YTickLabel', round(pYV.y([ymn ymx])*100)/100,'FontSize', 18)
    set(gca,'YTick',pYV.y([ymn ymx]))
    set(gca,'YTickLabel', round(pYV.y([ymn ymx])*100)/100,'FontSize', 18)       
    set(gca,'XTick', 1-pYV.v([vmn vmx]))
    set(gca,'xTickLabel', 1-round(pYV.v([vmn vmx])*100)/100, 'FontSize', 18)     
    %print('-f3','-depsc2', [outdir '/' bn '_yv1']);     
    
    %ms = min(size(pYV.f2));
    %pYV.f2 = imresize(imresize(pYV.f2, [ms ms], 'bilinear'), 4)';
    %pYV.f2(1:round(size(pYV.f2, 1)/10), round(9*size(pYV.f2, 2)/10):end) = mf;    
    %imwrite(pYV.f2, [outdir '/' bn '_initYV.jpg'], 'Quality', 100);
                 
    % WRITE FINAL VIEWPOINT PROB      
    %finalYV = reshape(marginal(1).T, numel(pYV.y), numel(pYV.v))'/max(marginal(1).T);   
    figure(4), surf(1-pYV.v(vmn:vmx),pYV.y(ymn:ymx), mYV2(ymn:ymx,vmn:vmx)/znorm), axis vis3d %axis square, axis on
    set(gca, 'ZGrid', 'off'), grid off, shading interp, view(-60, 30)
    set(gca,'ZTick',[pmn pmx])
    set(gca,'YTick',pYV.y([ymn ymx]))
    set(gca,'YTickLabel', round(pYV.y([ymn ymx])*100)/100,'FontSize', 18)       
    set(gca,'XTick', 1-pYV.v([vmn vmx]))
    set(gca,'xTickLabel', 1-round(pYV.v([vmn vmx])*100)/100, 'FontSize', 18)        
    
%    return;
%    disp('Waiting to adjust axes')
    
    print('-f3','-djpeg90', [outdir '/' bn '_yv1']);
    print('-f4','-djpeg90', [outdir '/' bn '_yv2']);
    
    
%     mf = max(marginal(1).T);
%     ms = min(size(mYV2)); 
%     mYV2 = imresize(imresize(mYV2, [ms ms], 'bilinear'), 4)';
%     mYV2(1:round(size(mYV2, 1)/10), round(9*size(mYV2, 2)/10):end) = mf;
%     imwrite(mYV2, [outdir '/' bn '_finalYV.jpg'], 'Quality', 100);
    
    % WRITE CONTEXT MAPS
    loadData = load([basedir '/context/' strtok(fn{f}, '.') '.c.mat']); 
    if 1 
    %for j = 1:8
    %    imwrite(loadData.cimages{1}(:, :, j), [outdir '/' bn '_c' num2str(j) '.jpg'], ...
    %        'Quality', 100);
    %end
    imwrite(loadData.cimages{1}(:, :, 1), [outdir '/' bn '_c1.jpg'], 'Quality', 90);
    imwrite(loadData.cimages{1}(:, :, 3), [outdir '/' bn '_c3.jpg'], 'Quality', 90);
    imwrite(sum(loadData.cimages{1}(:, :, [4 5 6 8]), 3), [outdir '/' bn '_c2.jpg'], ...
            'Quality', 90);            
    end
%     % WRITE OBJECT DETECTIONS   
%     threshi = [0.112 0.271]; % confidence thresholds for 2 FP/I (initial)
%     threshf = [0.102 0.172]; % confidence thresholds for 2 FP/I (final)    
%     %threshi = [0.144 0.224]; % confidence thresholds for 25% precision (initial)
%     %threshf = [0.271 0.350]; % confidence thresholds for 25% precision (final)
%               
%     figure(1), hold off, imshow(im), hold on
%     plot([1 size(im, 2)], [horiz(f) horiz(f)]*size(im, 1), 'b--', 'LineWidth', 3);
%     colors = 'gr';
%     for n = 1:ncand
%         if candidates(n).p > threshi(candidates(n).objType)
%             [tmp, ind] = max(candidates(n).conf);
%             %disp(num2str(candidates(n).p))
%             bbox = candidates(n).bbox(ind, :);
%             color = colors(candidates(n).objType);
%             plot(bbox([1 1 2 2 1]), bbox([3 4 4 3 3]), color ,'linewidth',3);
%         end
%     end    
%     saveas(1, [outdir '/' bn '_odinit.tif']);
%     
%     figure(2), hold off, imshow(im), hold on;        
%     
%     h2 = pHorizon(f).ml;
%     plot([1 size(im, 2)], [h2 h2]*size(im, 1), 'b--', 'LineWidth', 3);    
%     
%     count = 0;
%     for k = 2:ncand+1
%         pobj = sum(marginal(k).T(1:end-1));
%         if pobj > threshi(candidates(k-1).objType)
%             %disp(num2str(pobj))
%             [tmp, ind] = max(marginal(k).T(1:end-1));            
%             bbox = candidates(k-1).bbox(ind, :);
%             count = count + 1;
%             %c = colors(mod(count-1,6)+1);
%             c = colors(candidates(k-1).objType);
%             plot(bbox([1 1 2 2 1]), bbox([3 4 4 3 3]), c ,'linewidth',3);
%             %pause(1)
%         end
%     end    
%     saveas(2, [outdir '/' bn '_odfinal.tif']);
    if 1 % XXX
    % WRITE OBJECT PMAPS
    gtruth = ts.gtruth(ts.good_inds(f));

%    scales = 2^(1/4)*2.^([0:5]/2);    
%    scaleRange = 2^(1/4);
    
    scales = 2^(1/2)*2.^([0:2]);    
    scaleRange = 2^(1/2);    
    
    ot = 1;
    maxp = 0;
    clear gtind
    clear pim1 pim2
    for j = 1:numel(scales)
        
        w = bbox(2)-bbox(1);  h = bbox(4)-bbox(3);
        detsize = reshape([detector{ot}(:).objectSize]', 2, numel(detector{ot}))';
        %[tmp, v] = min(detsize(:, 1)./detsize(:, 2)-w/h);
        scale = detsize(:, 2)/h;
        h = h / size(im, 1);
        
        scale = scales(j);
        
        pim1{j} = zeros(ceil([size(im, 1) size(im, 2)]/scale));
        pim2{j} = zeros(ceil([size(im, 1) size(im, 2)]/scale));
        
        for k = 2:ncand+1
            
            if candidates(k-1).objType==ot
                
                margp1 = candidates(k-1).conf(:);%/maxp*sump;            
                               
                sump = sum(marginal(k).T(1:end-1));
                maxp = max(marginal(k).T(1:end-1))+(1E-10);
                margp2 = marginal(k).T(1:end-1)/maxp*sump;
                
                tx = ceil(mean(candidates(k-1).bbox(:, [1 2]), 2)/scale);
                ty = ceil(mean(candidates(k-1).bbox(:, [3 4]), 2)/scale);
                th = (candidates(k-1).bbox(:, 4)-candidates(k-1).bbox(:, 3))/mean(detsize(:, 2));
                ind = find( (th > scale/scaleRange) & (th < scale*scaleRange));
    
                for m = ind'
                    pim1{j}(ty(m), tx(m)) = max([pim1{j}(ty(m),tx(m)) margp1(m)]);
                    pim2{j}(ty(m), tx(m)) = max([pim2{j}(ty(m),tx(m)) margp2(m)]);
                end                        
                
            end                        
        end

        %pim1{j} = conv2(pim1{j}, fspecial('gaussian', [3 3], 1));
        %pim2{j} = conv2(pim2{j}, fspecial('gaussian', [3 3], 1));              

        maxp = max(maxp, max(pim1{j}(:)));
        maxp = max(maxp, max(pim2{j}(:)));
        
        th = (gtruth.bbox(:, 4)-gtruth.bbox(:, 3))/mean(detsize(:, 2));
        gtind{j} = find( (th > scale/scaleRange) & (th < scale*scaleRange) & (gtruth.objType'==ot));        
        
    end
    
    for j = 1:numel(pim1)
        pim1{j} = pim1{j} / maxp;
        pim1{j} = pim1{j}.^(1/3);
        pim1{j}(round(1:mean(detsize(:, 2))), round(1:mean(detsize(:, 1)))) = 0.5;              
        pim2{j} = pim2{j} / maxp;
        pim2{j} = pim2{j}.^(1/3);        
        pim2{j}(round(1:mean(detsize(:, 2))), round(1:mean(detsize(:, 1)))) = 0.5;         
        figure(5), hold off, imshow(pim1{j}), hold on
        plot(gtruth.bbox(gtind{j}, [1 1 2 2 1])'/scales(j), gtruth.bbox(gtind{j}, [3 4 4 3 3])'/scales(j), 'r', 'LineWidth', 3);
        print('-f5', '-djpeg90', [outdir '/' bn '_p1_' num2str(j) '.jpg']);
        figure(6), hold off, imshow(pim2{j}), hold on
        plot(gtruth.bbox(gtind{j}, [1 1 2 2 1])'/scales(j), gtruth.bbox(gtind{j}, [3 4 4 3 3])'/scales(j), 'r', 'LineWidth', 3);
        print('-f6', '-djpeg90', [outdir '/' bn '_p2_' num2str(j) '.jpg']);        
        %imwrite(pim1{j}.^(1/3), [outdir '/' bn '_p1_' num2str(j) '.jpg'], 'Quality', 90);
        %imwrite(pim2{j}.^(1/3), [outdir '/' bn '_p2_' num2str(j) '.jpg'], 'Quality', 90);
    end
%         m1 = max(pim1(:));
%         m2 = max(pim2(:));        
%         
%         pim1 = pim1 / m1;
%         pim2 = pim2 / m2;
%         
%         hf = size(im, 1)*mean(scale);
%         wf = mean(scale);
%         
%         % show box size
%         %pim1(1:round(h*hf), 1:round(w*wf)) = 0.5;
%         %pim2(1:round(h*hf), 1:round(w*wf)) = 0.5;
% 
%         % show confidence level
%         imw = round(max([size(pim1, 1) size(pim1, 2)])/10);
%         pim1(1:imw, end-imw+1:end) = m1;
%         pim2(1:imw, end-imw+1:end) = m2;           
%         
%         sf = 300/size(pim1, 2);
%         pim1 = repmat(imresize(pim1, sf, 'bilinear'), [1 1 3]);
%         pim2 = repmat(imresize(pim2, sf, 'bilinear'), [1 1 3]);
% 
%         %box = round([1 1 h*hf 1 ; 1 w*wf h*hf h*hf ; w*wf w*wf h*hf 1 ; 1 w*wf 1 1])';
%         figure(1), hold off, imshow(pim1);
%         box = bbox*mean(scale)*sf;
%         hold on, plot(box([1 1 2 2 1]), box([3 4 4 3 3]), color, 'LineWidth', 3)
%         saveas(1, [outdir '/' bn '_pobjinit' num2str(j) '.jpg']);
%         
%         figure(2), hold off, imshow(pim2);
%         box = bbox*mean(scale)*sf;
%         hold on, plot(box([1 1 2 2 1]), box([3 4 4 3 3]), color, 'LineWidth', 3)        
%         saveas(2, [outdir '/' bn '_pobjfinal' num2str(j) '.jpg']);
%         
%         %box = round(bbox([1 1 3 4 ; 1 2 4 4 ; 2 2 3 4 ; 1 2 3 3])')/mean(scale)*sf;
%         %pim1 = draw_line_image2(pim1, box, [1 0 0], 3);
%         %pim2 = draw_line_image2(pim2, box, [1 0 0], 3);        
%                      
%         %box = round([1 1 h*hf 1 ; 1 w*wf h*hf h*hf ; w*wf w*wf h*hf 1 ; 1 w*wf 1 1])';
%         %pim1 = draw_line_image2(pim1, box, 1);
%         %pim2 = draw_line_image2(pim2, box, 1);
% 
%         %imwrite(pim1, [outdir '/' bn '_pobjinit' num2str(j) '.jpg'], 'Quality', 100);
%         %imwrite(pim2, [outdir '/' bn '_pobjfinal' num2str(j) '.jpg'], 'Quality', 100);
%     end        

    end
    
    outdir = toutdir;
    
        %[tmp, scale] = min(abs(0.89^[0:50]*detsize(v, 2)-h))+1;
        
    %imshow(initYV, 'XData', pYV.y, 'YData', 1-pYV.v, 'InitialMagnification', 'fit'), axis square, axis on
    %figure(6), imshow(reshape(marginal(1).T, numel(pYV.y), numel(pYV.v))'/max(marginal(1).T), 'XData', pYV.y, 'YData', 1-pYV.v, 'InitialMagnification', 'fit'), axis square, axis on    
    
    drawnow;
    
    %disp(num2str([h2 pHorizon(f).ml]))
    %pause
    %newhorizon(f) = h2;
    %newcandidates{f} = candidates;

end