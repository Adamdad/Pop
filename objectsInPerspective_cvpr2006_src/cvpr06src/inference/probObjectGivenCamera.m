function cpt = probObjectGivenCamera(candidate, pYV, imsize)
% Get the conditional probability table for P( o | c, e)
% P(o | c, e) = P(o | e) P(h | t, v, c)
%
% Input:
%   candidate.(bbox, conf, p):
%       bbox(i, [x1 x2 y1 y2]): the bounding box of detection i
%       conf(i): the confidence of the object being at i given that the
%                candidate is the object
%       p: the overall confidence for the candidate
%       objtype: the type of object


conf = candidate.conf;

% cpt = P(obj | ev_obj) with last term being obj = background
cpt = [conf(:)'/sum(conf)*candidate.p (1-candidate.p)];
%cpt = cpt / sum(cpt);

% cpt = P(obj | ev_obj) * P(h | v,t,camera)
pObjGivenC = getCameraParameterProbs(candidate.bbox, pYV, candidate.objType, imsize);    

% if isfield(pYV, 'models_depth')
%     pObjGivenC = pObjGivenC .* getDepthProbs(candidate.bbox, pYV, candidate.objType, imsize);
% end

pObjGivenC(:, end+1) = 1;
cpt = repmat(cpt, size(pObjGivenC, 1), 1) .* pObjGivenC;    
 

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


% function yvMat = getDepthProbs(det, pYV, objtype, imsize)
% % get P(v | v0, yc) for each possible detection and each possible yc, v0
% % det(ndet, [x1 x2 y1 y2])
% 
% ndet = size(det, 1);
% 
% v = 1-(det(:, 4)-1)/imsize(1);
% h = (det(:, 4) - det(:, 3)) / imsize(1);
% 
% yvMat = zeros([size(pYV.f, 1) ndet], 'single'); 
% 
% modelt = pYV.models_depth(objtype);
% 
% for i = 1:ndet
%     
%     mu = -modelt.mu + log(1.36*pYV.ymat);
%     sigma = modelt.sigma; 
% 
%     dv = max(pYV.vmat-v(i), 0.0001);
%     yvMat(:, i) = single(exp(-(log(dv) - mu).^2 ./ (2*sigma.^2)) ./ ...
%         (dv * (sigma * sqrt(2) * pi)));
%            
% end