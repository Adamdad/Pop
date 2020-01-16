function [newcandidates, pHorizon, pHeight] = ...
    ovInference(candidates, imsize, pYV)
% [newcandidates, pHorizon, pHeight] = ...
%    govInference(pYV, pG, priorO, cimages, candidates)
% 
% Infers new marginals for object detections and viewpoint given object
% detection evidence, geometry evidence, and viewpoint priors.  
%
% Input: 
%   candidates - object detection candidates, with likelihoods
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
%   CPT: P(parent value, node value)

ncand = numel(candidates);

% count number of nodes and get node sizes
nnodes = 1 + ncand;
node_sizes = zeros(1, nnodes);
node_sizes(1) = numel(pYV.f);
for k = 1:ncand
    node_sizes(k+1) = numel(candidates(k).conf)+1; % object node
end

% create DAG and bayes net
dag = zeros(nnodes,nnodes);
dag(1, 2:(ncand+1)) = 1; % camera to object links
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


    

