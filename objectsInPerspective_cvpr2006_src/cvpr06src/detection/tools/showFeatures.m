function showFeatures(paramClassifier, dict)
% Shows the features selected by the classifier

NweakClassifiers = length(paramClassifier);

Nx = 5;

figure
for j = 1:NweakClassifiers
    f  = paramClassifier(j).featureNdx;
    
    subplot(NweakClassifiers, Nx, (j-1)*Nx + 1)
    imagesc(dict(f).wavelet); axis('equal'); axis('off'); axis('tight'); colormap(gray(256))
    
    subplot(NweakClassifiers, Nx, (j-1)*Nx + 2)
    imagesc(dict(f).fragment); axis('equal'); axis('off'); axis('tight'); colormap(gray(256))
    
    subplot(NweakClassifiers, Nx, (j-1)*Nx + 3)
    imagesc(conv2((dict(f).gaussianY)', dict(f).gaussianX)); axis('equal'); axis('off'); axis('tight'); colormap(gray(256))
    
    subplot(NweakClassifiers, Nx, (j-1)*Nx + 4)
    bar([dict(f).exp_wavelet dict(f).exponent],'g'); axis('tight'); %axis([.5 2.5 -1.1 1.1]); grid on
    
    subplot(NweakClassifiers, Nx, (j-1)*Nx + 5)
    bar([paramClassifier(j).a paramClassifier(j).b],'r'); axis('tight'); axis([.5 2.5 -1.1 1.1]); grid on
    
    drawnow
end


