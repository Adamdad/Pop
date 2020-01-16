function cpt = probGeomGivenObject(candidate, pG, cimages)
% Compute P(g | e, o) under conditional independence assumptions
% g has three possible values:
%    g=1: object region mostly not vert
%    g=2: obj mostly vert, below mostly not ground
%    g=3: obj mostly vert, below mostly ground
% vert is defined object-by-object
%
% INPUT:
% det(ndet, [conf x1 x2 y1 y2]); conf = log(p(o,I)/p(~o,I))
% pG.v1g1(ntypes): prob vertical over ground given object
%   .v1g0(ntypes): prob vertical over not ground given object
%   .bk_v1g1(ntypes): prob vertical over ground given not object
%   .bk_v1g0(ntypes): prob vertical over not ground given no object
%   .vind{ntypes}: defines which subclasses are considered vertical
% cimages: images of the confidence in the three vertical classes (1-3)
% and 5 subclasses (4-8) {gnd, vert, sky, left, center, right, por, sol}
% OUTPUT:
% cpt(objval, geomval): P(g | e) / P(g) P(g | o)

gconfim = cimages(:, :, 1);
for t = 1:numel(pG.vind)
    vconfim{t} = sum(cimages(:, :, 3+pG.vind{t}), 3);    
end

cpt = cell(size(candidate));

for k = 1:numel(candidate)

    % initialize for this object type
    t = candidate(k).objType;
    pGgivenO(1:3) = [1-pG.v1g1(t)-pG.v1g0(t) pG.v1g0(t)    pG.v1g1(t)];
    % note: practically speaking, this is uniform, since the first value will
    % always be multiplied by a zero.  This is a hack, that was first invented
    % by a benevolent bug.  
    prG(1:3) = [1-pG.bk_v1g1(t)-pG.bk_v1g0(t) pG.bk_v1g0(t) pG.bk_v1g1(t)];
    %prG(1:3) = [1-pG.bk_v1g1(t)-pG.bk_v1g0(t) pG.bk_v1g0(t) pG.bk_v1g0(t)];
    
    
    det = candidate(k).bbox;

    ndet = size(det, 1);

    % 1:ndet are for possible object positions, ndet+1 is for non-object
    cpt{k} = zeros(ndet+1, 3);

    x1 = round(max(min(det(:, 1), size(cimages, 2)), 1));
    x2 = round(max(min(det(:, 2), size(cimages, 2)), 1));
    y1 = round(max(min(det(:, 3), size(cimages, 1)), 1));
    y2 = round(max(min(det(:, 4), size(cimages, 1)), 1));

    by1 = y2+1;
    by2 = max(min(round(by1+(y2-y1)/2-0.5), size(cimages, 1)), 1);

    for i = 1:ndet

        b1 = [y1(i):y2(i)];  b2 = [x1(i):x2(i)];  
        pv = mean(mean(vconfim{t}(b1, b2)));

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

        cpt{k}(i, :) = pGgivenE ./ prG .* pGgivenO;

        if i==1
            cpt{k}(end, :) = pGgivenE;
        end

    end
end

%if numel(cpt)==1
%    cpt = cpt{1};
%end