function [newcandidates, pHorizon, pHeight] = ...
    govInference_margeCam(candidates, cimages, pG, pYV)
% [newcandidates, pHorizon, pHeight] = ...
%    govInference(pYV, pG, priorO, cimages, candidates)
% 
% Infers new marginals for object detections and viewpoint given object
% detection evidence, geometry evidence, and viewpoint priors.  This
% marginalizes over the camera for a specific experiment.
%
% Input: 
%   candidates - object detection candidates, with likelihoods
%   cimages - geometry estimates
%   pG - likelihood of geometry given object and of just geometry
%   pYV - prior for camera height (y) and horizon position (v)
% Output: 
%   newcandidates - object detection candidates, with updated likelihoods
%   pHorizon - marginal for horizon position
%   pHeight - marginal for camera height
%
% Notes on Bayesian network model:
%   total nodes: 2*n_o+1 where n_o is num objects
%   node 1: camera state - values specify pair camera height, horizon       
%   node 2 to n_o+2: object state - last value specifies background,
%       value j=1..n_o specifies object presence at location (j-1)
%   node n_o+3:2*n_o+2: local geometry state corresponding to nodes
%       2 to n_o+1 before - values 1,2,3,4 represent notObjSurf/notGndBel,
%       notObjSurf/gndBel, objSurf/notGndBel, objSurf/gndBel
%   CPT: P(parent value, node value)

ncand = numel(candidates);
imsize = [size(cimages, 1) size(cimages, 2)];

% count number of nodes and get node sizes
nnodes = 1 + 2*ncand;
node_sizes = zeros(1, nnodes);
node_sizes(1) = numel(pYV.f);
for k = 1:ncand
    node_sizes(k+1) = numel(candidates(k).conf)+1; % object node
    node_sizes(k+1+ncand) = 3; % geom node
end

% create DAG and bayes net
dag = zeros(nnodes,nnodes);
%dag(1, 2:(ncand+1)) = 1; % camera to object links
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
    cpt = sum(cpt .* repmat(pYV.f, [1 size(cpt, 2)]), 1)';
    bnet.CPD{k+1} = tabular_CPD(bnet, k+1, 'CPT', cpt);
end

% geometry
cpt = probGeomGivenObject(candidates, pG, cimages);
for k = 1:ncand    
    bnet.CPD{ncand+k+1} = tabular_CPD(bnet, ncand+k+1, 'CPT', cpt{k});
end

% do marginalization
evidence = cell(1, nnodes);
engine = pearl_inf_engine(bnet, 'protocol', 'tree');
engine = enter_evidence(engine, evidence, 'maximize', 0);
for n = 1:nnodes
    marginal(n) = marginal_nodes(engine, n);
end

% create new candidates structure
newcandidates = candidates;
for n = 2:(ncand+1)
    pobj = sum(marginal(n).T(1:end-1));
    newcandidates(n-1).p = pobj;
    newcandidates(n-1).conf = marginal(n).T(1:end-1);
end

% get horizon and camera height marginals
% do 1-val so that 0 is the top of the image 
if nargout > 1
    pYV.vmat = 1-pYV.vmat;
    pYV.v = 1-pYV.v;
    [pHorizon, pHeight] = cameraJointToMarginals(pYV, marginal(1).T);    
end


    

