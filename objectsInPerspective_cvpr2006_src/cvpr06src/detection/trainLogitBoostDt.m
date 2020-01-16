function classifier = trainLogitBoostDt(features, labels, ntrees, nnodes, prior)
% classifier = trainLogitBoostDt(features, labels, ntrees, nnodes)
% Trains a 2-class classifier and learns probabilistic output parameters using
% cross-validation.
% 
% Input:
%   features(ndata, nfeatures)
%   labels(ndata): in {-1, 1}
%   ntrees: number of decision trees
%   nnodes: number of nodes per tree
%   prior: expected number of objects per candidate
% Output:
%   classifier: boosted decision tree classifier with probabilistic output
%               parameters

LOAD = 0;
DO_CV = 0; % for cross-validation

% re-order features and labels so that positive come first
posind = find(labels==1);
negind = find(labels==-1);
features = [features(posind, :) ; features(negind, :)];
labels = [labels(posind, :) ; labels(negind, :)];

np = length(posind);
nn = length(negind);

if DO_CV

    if LOAD
        disp('loading saved data!')
        load '../data/tmpdatacar_tv1.mat'
    else
    
        % randomize indices for cross-validation
        rpind = randperm(np);
        rnind = np+randperm(nn);

        ncv = 3;

        % create classifier for each round of cross-validation and evaluate
        % remaining data
        cvconf = zeros(size(labels));
        for v = 1:ncv
            testind = [rpind(round(np/ncv*(v-1))+1:round(np/ncv*v))  ...
                rnind(round(nn/ncv*(v-1))+1:round(nn/ncv*v))];
            trainind = setdiff([1:np+nn], testind);
            tmpclassifier =  train_boosted_dt_2c(features(trainind, :), ...
                [], labels(trainind), ntrees, nnodes);
            cvconf(testind) = testBoostedDtMc(tmpclassifier, features(testind, :));
        end

        cvconf = 1 ./ (1+exp(-cvconf));
        save '../data/tmpdatacar_tv1.mat' cvconf;
    end

    % get probabilistic output parameters (not needed since logitboost does a
    % good job by itself, as I found by experiments)
    xvals = [0.01:0.02:0.99];  
    p0 = hist(cvconf(np+1:end), xvals)+1E-10; % P(conf | y=-1)
    p1 = hist(cvconf(1:np), xvals)+1E-10; % P(conf | y=1)
    p0 = p0 / sum(p0) * 0.99;
    p1 = p1 / sum(p1) * 0.01;
    f = p1 ./ (p0+p1);

    [A, B, err] = getProbabilisticOutputParams(log(xvals ./ (1-xvals)), f)

    [sconf, sind] = sort(cvconf, 'descend');
    tp = cumsum(labels(sind)==1);
    fp = cumsum(labels(sind)==-1);
    figure(1), hold off, plot(fp / max(fp), tp / max(tp))
    drawnow; pause(1)

end
    
% if 1
%     figure(1), hold off %, plot(xvals, p0 / (1-prior), 'r', 'LineWidth', 2);
%     %hold on, plot(xvals, p1 / prior, 'g', 'LineWidth', 2);
%     hold off, plot(xvals, (p1 / prior) ./ (p1/prior + p0/(1-prior)), 'r', 'LineWidth', 2);
%     plot(xvals, f, '+b', 'LineWidth', 2);
%     plot(xvals, 1./(1+exp(log(xvals ./ (1-xvals))*A+B)), 'b', 'LineWidth', 2);
%     drawnow;
%     pause(1);
% end 

% train classifier with full dataset
classifier = train_boosted_dt_2c(features, [], labels, ntrees, nnodes);

%classifier.poparams = [A B];
classifier.type = 'boostDT';  
if DO_CV
    classifier.cvroc.tp = tp;
    classifier.cvroc.fp = fp;
end
classifier.objectSize = [];
