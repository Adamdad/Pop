function pYV = cameraInitYV(pY, pV)
% pYV = cameraInitYV(pV, pY)
% Returns P(yc, v0) = P(yc)P(v0) 
%
% INPUT
% pY - "y" - the yc values, "f" - the prior density
% pV - "v" - the v0 values, "f" - the prior density
%
% OUTPUT
% pYV = "y" - the yc values, "v" the v0 values, "f" the joint density

nv = length(pV.x);
ny = length(pY.x);

pYV.y = pY.x;
pYV.v = pV.x;
pYV.f = reshape(repmat(pY.f, 1, nv) .* repmat(pV.f', ny, 1), [ny*nv 1]);

pYV.f = pYV.f / sum(pYV.f(:));
pYV.ymat = reshape(repmat(pYV.y, 1, nv), [ny*nv 1]);
pYV.vmat = reshape(repmat(pYV.v', ny, 1), [ny*nv 1]);

pYV.maxf = max(pYV.f);

% old (CVPR'06 models)
% car
pYV.models(1).mu = 1.59;
pYV.models(1).sigma = 0.21;

% pedestrian (adult)
pYV.models(2).mu = 1.7;
pYV.models(2).sigma = 0.085;

% new (IJCV models, from labelme)
% car
pYV.models(1).mu = 1.51;
pYV.models(1).sigma = 0.191;

% pedestrian (adult)
pYV.models(2).mu = 1.7;
pYV.models(2).sigma = 0.103;

% object depth model
% pYV.models_depth(1).mu = 3.3;
% pYV.models_depth(1).sigma = 0.71;
% pYV.models_depth(2).mu = 2.77;
% pYV.models_depth(2).sigma = 0.72;


