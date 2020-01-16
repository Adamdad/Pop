function classifier = gentleBoost(x, y, Nrounds, varargin)
% gentleBoost
%
% features x
% class: y = [-1,1]
%
% modified by Derek Hoiem 2005 to admit binary values

% Implementation of gentleBoost:
% Friedman, J. H., Hastie, T. and Tibshirani, R. 
% "Additive Logistic Regression: a Statistical View of Boosting." (Aug. 1998) 

% atb, 2003

maxAfterZero = Nrounds;
if length(varargin)>0
    maxAfterZero = varargin{1};
end

[Nfeatures, Nsamples] = size(x); % Nsamples = Number of thresholds that we will consider
Fx = zeros(1, Nsamples);
w  = ones(1, Nsamples); %w = w/sum(w);

pind = find(y==1);
nind = find(y==-1);
initposval = 1/2 * (1/length(pind));
initnegval = 1/2 * (1/length(nind));
w(pind) = initposval;
w(nind) = initnegval;
initw = w;

nAfterZero = 0;

% whether a feature is discrete
isb = zeros(Nfeatures, 1);
for f = 1:Nfeatures
    isb(f) = all(x(f, :)==0 | x(f, :)==1);
end

for m = 1:Nrounds
    disp(sprintf('Round %d', m))
    
    % weak regression 
    [featureNdx, th, a , b] = selectBestRegressionStump(x, y, w, isb);
    disp(['feature: ' num2str(featureNdx)])
    
    % update parameters classifier
    classifier(m).featureNdx = featureNdx;
    classifier(m).th = th;
    classifier(m).a  = a;
    classifier(m).b  = b;
    
    % Updating and computing classifier output
    fm = (a * (x(featureNdx,:)>th) + b);
    Fx = Fx + fm;
    
    w1 = sum(w);
    w = w .* exp(-y.*fm);
    disp(['Z: ' num2str(sum(w)/w1)]);
    err = mean(w>initw);
    disp(['error (p n all): ' ...
        num2str([mean(w(find(y==1))>initposval) mean(w(find(y==-1))>initnegval)  err])]);
    pause(0.01)
    
    if err == 0        
        nAfterZero = nAfterZero + 1;
        if nAfterZero > maxAfterZero
            break;
        end
    end               
    
    %w = w / sum(w);
end

