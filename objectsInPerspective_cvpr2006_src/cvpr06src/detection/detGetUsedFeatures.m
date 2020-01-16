function used = detGetUsedFeatures(classifier)
% used = detGetUsedFeatures(classifier)
% Returns whether or not each of the possible features are used by the
% given classifier.

nfeatures = classifier.wcs(1).dt(1).npred;
used = zeros(nfeatures, 1);

nwc = numel(classifier.wcs);
for w = 1:nwc
    wc = classifier.wcs(w);
    ndt = numel(wc.dt);
    for t = 1:ndt
        uf = abs(wc.dt.var(find(abs(wc.dt.var)>0)));
        used(uf) = used(uf) + 1;
    end
end
        