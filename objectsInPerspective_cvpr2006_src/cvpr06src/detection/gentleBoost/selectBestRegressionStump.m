function [featureNdx, th, a , b, error] = selectBestRegressionStump(x, z, w, isb);
% [th, a , b] = fitRegressionStump(x, z);
% z = a * (x>th) + b;
%
% where (a,b,th) are so that it minimizes the weighted error:
% error = sum(w * |z - (a*(x>th) + b)|^2) / sum(w)
% isb - whether a feature is binary

% atb, 2003
% torralba@ai.mit.edu
% modified by Derek Hoiem 2005 to allow binary features

[Nfeatures, Nsamples] = size(x); % Nsamples = Number of thresholds that we will consider
w = w/sum(w); % just in case...

th = zeros(1,Nfeatures);
a = zeros(1,Nfeatures);
b = zeros(1,Nfeatures);
error = zeros(1,Nfeatures);

for n = 1:Nfeatures
    [th(n), a(n) , b(n), error(n)] = fitRegressionStump(x(n,:), z, w, isb(n));
end

[error, featureNdx] = min(error);
th = th(featureNdx);
a = a(featureNdx);
b = b(featureNdx);
