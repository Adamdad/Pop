function cvprIntroFigure(fn, objtype, outdir, trueh, gtboxes, gttype)
% cvprIntroFigure(fn, objtype, outdir, gtbox, gttype)


% INITIALIZATION
basedir = '/IUS/vmr20/dhoiem/datasets/testset';
imdir = [basedir '/allimages'];
contextdir = [basedir '/context'];
objdir = [imdir '/objectDetections'];
datadir = '/IUS/vmr7/dhoiem/context/objdet/data';
loadData = load([datadir '/labelme_cameraHeightPrior.mat']);
priorY = loadData.priorY;
loadData = load([datadir '/labelme_horizprob2.mat']);
priorV_tmp = loadData.pH;
priorV.x = 1-priorV_tmp.x; % + (1-horiz(f));
priorV.f = priorV_tmp.f;
pYV = cameraInitYV(priorY, priorV);  
    
loadData = load([datadir '/pG_car_ppl.mat']);
pG = loadData.pG;

objboxes = [32 24; 16 40];
objbox = objboxes(objtype, :);
colors = 'gr';
color = colors(objtype);

for f = 1:numel(fn)

    close all
    
    bn = strtok(fn{f}, '.');
    tmpdir = [outdir '/' bn];
    if ~exist(tmpdir)
        mkdir(outdir, bn);
    end        
           
    im = imread([imdir '/' fn{f}]);
    %im = imresize(im, 0.5);
    [imh, imw, imb] = size(im);
    
    % NORMAL IMAGE
    figure(1), hold off, imshow(im);
    saveas(1, [tmpdir '/' fn{f}])    
    
    % RANDOM BOXES ON IMAGE
    scales = 1./(0.89.^[0:100]);
    maxscale = max(find(min(imh / objbox(2), imw / objbox(1))>scales));
    nboxes = 15000;
    rs = scales(ceil(rand(nboxes, 1)*maxscale))';    
    rx = ceil(rand(nboxes, 1).*(imw-rs*objbox(1))+rs*objbox(1)/2);
    ry = ceil(rand(nboxes, 1).*(imh-rs*objbox(2))+rs*objbox(2)/2);
    boxes = [repmat(rx, 1, 2)+rs*[-0.5 0.5]*objbox(1) ...
        repmat(ry,1,2)+rs*[-0.5 0.5]*objbox(2)];

    boxinds = ceil(rand(1, 100)*nboxes);  
    figure(1), hold off, imshow(im), hold on, 
    plot(boxes(boxinds, [1 2 2 1 1])', boxes(boxinds,[4 4 3 3 4])', 'r', 'LineWidth', 3);        
    pause(1)
    saveas(1, [tmpdir '/' bn '_1.tif'])
    
    % HORIZON GIVEN OBJECTS
    yvMat = [];
    for t = 1:2
        ind = find(gttype{f}==t);
        if ~isempty(ind)
            yvMat = [yvMat getCameraParameterProbs(gtboxes{f}(ind, :), pYV, t, [imh imw])];
        end
    end
    pYVgO = pYV.f.*prod(yvMat, 2);
    pYVgO = pYVgO / sum(pYVgO);
    % get the 90% confidence bounds for the horizon
    fv = sum(reshape(pYVgO, [numel(pYV.y) numel(pYV.v)]), 1);        
    [tmp, maxind ] = max(fv);
    cumsumfv = cumsum(fv);
    [tmp, botind] = min(abs(cumsumfv(maxind) - cumsumfv - 0.45));
    [tmp, topind] = min(abs(cumsumfv - cumsumfv(maxind) - 0.45));
    maxv = pYV.v(maxind);  botv = pYV.v(botind);  topv = pYV.v(topind);                
    % get the 90% confidence bounds for the camera height
    fy = sum(reshape(pYVgO, [numel(pYV.y) numel(pYV.v)]), 2);        
    [tmp, maxind ] = max(fy);
    cumsumfy = cumsum(fy);
    [tmp, botind] = min(abs(cumsumfy(maxind) - cumsumfy - 0.45));
    [tmp, topind] = min(abs(cumsumfy - cumsumfy(maxind) - 0.45));
    maxy = pYV.y(maxind);  boty = pYV.y(botind);  topy = pYV.y(topind);    
    figure(2), hold off, imshow(im), hold on;
    plot([1 imw], (1-maxv)*imh+1*ones(1, 2), 'b-', 'linewidth', 3);
    plot([1 imw], (1-botv)*imh+1*ones(1, 2), 'b--', 'linewidth', 2);        
    plot([1 imw], (1-topv)*imh+1*ones(1, 2), 'b--', 'linewidth', 2);         
    if ~isempty(gtboxes)
        plot(gtboxes{f}(:, [1 1 2 2 1])', gtboxes{f}(:, [3 4 4 3 3])', 'k', 'linewidth', 3);
    end
    disp([maxy maxv])
    boty = round(boty*10)/10;
    topy = round(topy*10)/10;
    annotation('textbox', [0.4 0.1 0.2 0.08], ...
            'string', [num2str(boty) '<y_c<' num2str(topy)], ...
            'linestyle', 'none', 'color', 'w', 'fontsize', 24, 'background', 'k')    ;
    pause(1);
    saveas(2, [tmpdir '/' bn '_2.eps'])    
    
    % OBJECTS GIVEN CAMERA   
    givenPV.x = maxv;%1-trueh(f);
    givenPV.f = 1;
    givenPY.x = maxy;%2;
    givenPY.f = 1;
    givenPYV = cameraInitYV(givenPY, givenPV);
    yvMat = getCameraParameterProbs(boxes, givenPYV, objtype, [imh imw]);
    pObjC = sum(yvMat .* repmat(givenPYV.f, 1, nboxes), 1);
    pObjcum = cumsum(pObjC);
    pObjcum = pObjcum / pObjcum(end);
%     [vals, sind] = sort(pObjC, 'descend');    
%     boxinds = sind(1:10);       
    boxinds = zeros(100, 1);
    for k = 1:numel(boxinds)
        boxinds(k) = min(find(rand(1) < pObjcum));
    end
    figure(3), hold off, imshow(im), hold on, 
    plot(boxes(boxinds, [1 2 2 1 1])', boxes(boxinds,[4 4 3 3 4])', 'r', 'LineWidth', 3);     
    drawnow;    
    pause(1);
    saveas(3, [tmpdir '/' bn '_3.tif'])
    
    % OBJECTS GIVEN GEOMETRY
    loadData = load([basedir '/context/' strtok(fn{f}, '.') '.c.mat']);
    figure(4), imshow(loadData.cimages{1}(:, :, [2 1 3]))
    drawnow;
    pause(1)
    imwrite(loadData.cimages{1}(:, :, [2 1 3]), [tmpdir '/' bn '_4.png']);
    %saveas(4, [tmpdir '/' bn '_4.tif'])       
    
    
    [tmp, maxlabel] = max(loadData.cimages{1}(:, :, [1 2 3]), [], 3);
    %loadData.cimages{1}(:, :, [1 8 3]) = 0;
    tmpimages = loadData.cimages{1};
    loadData.cimages{1}(:, :, 1) = (tmpimages(:, :, 1) > tmpimages(:, :, 2)) & ...
        (tmpimages(:, :, 1) > tmpimages(:, :, 3));
    loadData.cimages{1}(:, :, 8) = (tmpimages(:, :, 2) > tmpimages(:, :, 1)) & ...
        (tmpimages(:, :, 2) >tmpimages(:, :, 3));    
    loadData.cimages{1}(:, :, 3) = (tmpimages(:, :, 3) > tmpimages(:, :, 2)) & ...
        (tmpimages(:, :, 3) > tmpimages(:, :, 1));        
        
%     maxlabel = maxlabel + (6*(maxlabel==2); % go from 2 to 8
%     for k = 1:numel(loadData.cimages{1})
%         loadData.cimages{1}(:, :, maxlabel(k)) = 1;
%     end
    
%     loadData.cimages{1} = loadData.cimages{1}.^5;
%     loadData.cimages{1} = loadData.cimages{1} ./
    %loadData.cimages{1} = imresize(loadData.cimages{1}, 0.5, 'bilinear');
%     loadData.cimages{1}(:, :, 1) = 0;
%     loadData.cimages{1}(round(end/2)+1:end, :, 1) = 1;
%     loadData.cimages{1}(:, :, 8) = 0;  
%     loadData.cimages{1}(end/4+1:end/2,:, 8) = 1;
%     loadData.cimages{1}(:, :, 3) = 0;
%     loadData.cimages{1}(1:end/4, :, 3) = 1;      
    pObjG = probGeomGivenObject(boxes, objtype, pG, loadData.cimages{1});
    pObjG = sum(pObjG, 2).^5;
%     [tmp, ind] = max(pObjG);
%     [vals, sind] = sort(pObjG, 'descend');
    pObjGcum = cumsum(pObjG);
    pObjGcum = pObjGcum / pObjGcum(end);    
    boxinds = zeros(100, 1);
%     boxinds = sind(1:10);
    for k = 1:numel(boxinds)
        boxinds(k) = min(find(rand(1) < pObjGcum));
    end
    
    figure(5), hold off, imshow(im), hold on, 
    plot(boxes(boxinds, [1 2 2 1 1])', boxes(boxinds,[4 4 3 3 4])', 'r', 'LineWidth', 3);    
%     plot(boxes(ind, [1 2 2 1 1])', boxes(ind,[4 4 3 3 4])', 'k', 'LineWidth', 3);  
    pause(1)
    saveas(5, [tmpdir '/' bn '_5.tif'])   
    
    % OBJECTS GIVEN GEOMETRY AND CAMERA
    pObjC = pObjC / sum(pObjC);
    pObjG = pObjG / sum(pObjG);
    pObjCG = pObjC'.*pObjG;
    pObjCG = cumsum(pObjCG);
    pObjCG = pObjCG / pObjCG(end); 
%     [vals, sind] = sort(pObjCG, 'descend');    
%     boxinds = sind(1:10);    
   boxinds = zeros(100, 1);

    for k = 1:numel(boxinds)
         boxinds(k) = min(find(rand(1) < pObjCG));
    end      
    figure(6), hold off, imshow(im), hold on, 
    plot(boxes(boxinds, [1 2 2 1 1])', boxes(boxinds,[4 4 3 3 4])', 'r', 'LineWidth', 3);     
    pause(1)
    saveas(6, [tmpdir '/' bn '_6.tif'])     
    %plot(boxes(boxinds(1), [1 2 2 1 1])', boxes(boxinds(1),[4 4 3 3 4])', 'k', 'LineWidth', 3);
end
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function yvMat = getCameraParameterProbs(det, pYV, objtype, imsize)
% get P(h | v0, yc, v) for each possible detection and each possible yc, v0
% det(ndet, [x1 x2 y1 y2])

ndet = size(det, 1);
v = 1-(det(:, 4)-1)/imsize(1);
h = (det(:, 4) - det(:, 3)) / imsize(1);
yvMat = single(zeros([size(pYV.f, 1) ndet])); 
modelt = pYV.models(objtype);

for i = 1:ndet    
    mu = modelt.mu * (pYV.vmat-v(i)) ./ pYV.ymat;
    sigma = modelt.sigma * (pYV.vmat-v(i)) ./ pYV.ymat;
    sigma = max(sigma, 1E-10);
    yvMat(:, i) = single(exp(-0.5*((h(i)-mu)./sigma).^2) ./ (sqrt(2*pi)*sigma));    
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function cpt = probGeomGivenObject(det, objtype, pG, cimages)

% initialize for this type of object
vconfim = sum(cimages(:, :, 3+pG.vind{objtype}), 3);
gconfim = cimages(:, :, 1);

pGgivenO(1:3) = [1-pG.v1g1(objtype)-pG.v1g0(objtype)    ...
    pG.v1g0(objtype)    pG.v1g1(objtype)];
prG(1:3) = [1-pG.bk_v1g1(objtype)-pG.bk_v1g0(objtype)    ...
    pG.bk_v1g0(objtype) pG.bk_v1g1(objtype)];

ndet = size(det, 1);

cpt = zeros(ndet, 3);

x1 = round(max(min(det(:, 1), size(cimages, 2)), 1));
x2 = round(max(min(det(:, 2), size(cimages, 2)), 1));
y1 = round(max(min(det(:, 3), size(cimages, 1)), 1));
y2 = round(max(min(det(:, 4), size(cimages, 1)), 1));

by1 = y2+1;
by2 = max(min(round(by1+(y2-y1)/2-0.5), size(cimages, 1)), 1);

for i = 1:ndet

    b1 = [y1(i):y2(i)];  b2 = [x1(i):x2(i)];  
    pv = mean(mean(vconfim(b1, b2)));
    pGgivenE(1) = 1-pv;    
    if y2(i)==size(cimages, 1) % nothing can be seen below object region
        % vertical object region but below object region cannot be seen
        pGgivenE(2) = pv * prG(2)/sum(prG(2:3)); % P(g=vrt|e)P(g=~gnd|g=vrt)
        pGgivenE(3) = pv * prG(3)/sum(prG(2:3)); % P(g=vrt|e)P(g=gnd|g=vrt)   
    else        
        b1 = [by1(i):by2(i)];  b2 = [x1(i):x2(i)];         
        pg = mean(mean(gconfim(b1,b2)));
        pGgivenE(2) = pv * (1-pg);
        pGgivenE(3) = pv * pg;              
    end    
    cpt(i, :) = pGgivenE ./ prG .* pGgivenO;    

end