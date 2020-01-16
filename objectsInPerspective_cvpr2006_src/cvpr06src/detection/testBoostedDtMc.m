function confidences = testBoostedDtMc(classifier, features)
% returns a confidence for each class in the classifier    

wcs = classifier.wcs;  
ntrees = size(wcs, 1);
nclasses = size(wcs, 2);
% wcs = fitrtree(wcs);
confidences = zeros(size(features, 1), nclasses);
for c = 1:nclasses
    for t = 1:ntrees
        [class_indices, nodes, classes] = treeval(wcs(t, c).dt, features);        
%          [class_indices, nodes, classes] = predict(wcs(t, c).dt, features);    
        confidences(:, c) = confidences(:, c) + wcs(t, c).confidences(nodes);
    end
    confidences(:, c) = confidences(:, c) + classifier.h0(c);
end
