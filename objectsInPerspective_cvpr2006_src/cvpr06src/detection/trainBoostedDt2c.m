function classifier = trainBoostedDt2c(features, cat_features, ...
    labels, num_iterations, nodespertree)
% Train a two-class classifier based on boosted decision trees.  Boosting done by the
% logistic regression version of Adaboost (Adaboost.L - Collins, Schapire,
% Singer 2002).  At each iteration, a set of decision trees is created, with
% confidences equal to 1/2*ln(P+/P-), according to the
% weighted distribution.  Weights are assigned as
% w(i,j) = 1 / (1+exp(sum{t in iterations}[yij*ht(xi, j)])).  
% features(ndata, nfeatures)
% cat_fatures - discrete-valued output feature indices
% labels - {-1, 1}
% num_iterations - the number of trees to create



num_data = length(labels);

cl = [-1 1];
y = labels;
w = zeros(num_data, 1);

for i = 1:2
    indices = find(y==cl(i));
    count = numel(indices);
    if cl(i)==1
        classifier.h0 = log(count / (num_data-count));
    end
    w(indices) = 1 / count/2;
end    
data_confidences = zeros(num_data, 1);

for t = 1:num_iterations
    % learn decision tree based on weighted distribution
    dt = treefitw(features, y, w, 1/num_data/2, 'catidx', cat_features, 'method', 'classification', 'maxnodes', nodespertree*4);
    [tmp, level] = min(abs(dt.ntermnodes-nodespertree));
    dt = treeprune(dt, 'level', level-1);
    % assign partition confidences
    pi = (strcmp(dt.classname{1},'1')) + (2*strcmp(dt.classname{2},'1'));
    ni = (strcmp(dt.classname{1},'-1')) + (2*strcmp(dt.classname{2},'-1'));
    confidences = 1/2*(log(dt.classprob(:, pi)) - log(dt.classprob(:, ni)));             

    % assign weights
    [class_indices, nodes, classes] = treeval(dt, features);        
    data_confidences = data_confidences + confidences(nodes);
    w = 1 ./ (1+exp(y.*data_confidences));        
    w = w / sum(w);
           
    disp(['c: ' num2str(mean(1 ./ (1+exp(-y.*data_confidences)))) '  e: ' num2str(mean(y.*data_confidences < 0))]);  
    
    classifier.wcs(t,1).dt = dt;
    classifier.wcs(t,1).confidences = confidences;       
    pause(0.1);
end

disp(['mean conf = ' num2str(mean(1 ./ (1+exp(-y.*data_confidences))))]);
disp(['training error: ' num2str(mean(y.*data_confidences < 0))]);    






        