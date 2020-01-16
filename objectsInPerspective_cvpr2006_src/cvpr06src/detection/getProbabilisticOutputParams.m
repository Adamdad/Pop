function [A, B, err] = getProbabilisticOutputParams(conf, labels)
% [A, B, err] = getProbabilisticOutputParams(conf, labels)
%
% Converts a score (such as SVM output or log-likelihood ratio) to a
% probability.
%
% Input:
%   conf(ndata): the confidence of a datapoint (higher indicates greater
%   likelihood of label(i)=1
%   label(ndata): the true label {0,1} of the datapoint
% Output:
%   A, B: p = 1 / (1+exp(A*conf+B))
%   err: final value that has been minimized

AB = fminsearch(@(AB) logisticError(AB, conf, labels), [-1 8], []);%, optimset('MaxFunEvals', 1000000, 'MaxIter', 1000000));

A = AB(1);
B = AB(2);

err = logisticError([A B], conf, labels)/numel(labels);


function err = logisticError(AB, conf, labels)

p = 1./ (1+exp(AB(1)*conf+AB(2)));

err = -sum(labels.*log(p)+(1-labels).*log(1-p));



