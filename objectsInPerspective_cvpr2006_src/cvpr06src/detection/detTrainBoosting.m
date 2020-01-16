function detector = detTrainBoosting(imdir, detParams, data, gtruth, ...
    dictionary, featuresTrain, classesTrain)
% 
% Input:
% imdir - the location of the images        
% detparams.percTrainingImages: proportion of images to be used for 
%                               training (suggest 0.8)
%           NweakClassifiers: (suggest 100)
%           objectSize: size of object window
% data: the data from tod_createData
% dictionary: the data from tod_createDictionary
% featuresTrain: the data from tod_computeFeatures
% classesTrain: the class labels from tod_computeFeatures
%
% Output:
% detector.classifier: the classifier
% detector also has the necessary data and parameters attached
                     

%% PARAMETERS
percTrainingImages = detParams.percTrainingImages;
NweakClassifiers = detParams.NweakClassifiers; %(100)
objectSize = detParams.objectSize;


Nclasses = 1;

 
Nfeatures = size(featuresTrain, 2);

Nimages = size(featuresTrain, 1);
size(classesTrain)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Select training and test images and sample the features: 
jTrain = randperm(Nimages); 
N = fix(Nimages*percTrainingImages)
sets.jTrain = jTrain(1:N);
sets.jTest  = setdiff(1:Nimages, sets.jTrain);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the features for training:
jTrain = sets.jTrain;
Nimages = length(jTrain);


ndx = 0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TRAINING THE JOINT DETECTORS:


classifier = gentleBoost(featuresTrain', (2*classesTrain-1)', NweakClassifiers, 20);
% Show the first 10 features selected
%showFeatures(classifier(1:10), dictionary)

detector.classifier = classifier;
detector.dictionary = dictionary;
detector.sets = sets;
detector.NweakClassifiers = NweakClassifiers;
detector.objectSize = objectSize;
detector.data = data;




if (0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% APPLY DETECTOR ON TEST SET:

%jTest = sets.jTest;
jTest = sets.jTrain;
names = sets.names;
Nimages = length(jTest);
d = dictionary;
for i = 1:Nimages
%    i
    name = names(jTest(i)).name;
  
    %nameFeat = [folder name];
    %nameImg = strrep(nameFeat,'_feat.mat','_image.jpg');
    nameImg = str2num(strrep(name, '_feat.mat', ''));
    
    img = im2double(imread([imdir '/' gtruth(nameImg).imname]));
    if size(img, 3)==3
        img = rgb2gray(img);
    end
    
    
    score=0;
    feature = zeros([size(img,1) size(img,2) NweakClassifiers]);
    for j = 1:NweakClassifiers
        f  = classifier(j).featureNdx;
        a  = classifier(j).a;
        b  = classifier(j).b;
        th = classifier(j).th;
        
        feature(:,:,j) = convCrossConv(img, d(f).wavelet, d(f).fragment, ...
				       d(f).gaussianY, d(f).gaussianX, ...
				       d(f).exp_wavelet, d(f).exponent);  
        score = score + (a * (feature(:,:,j) > th) + b);
    end
    
    figure
    subplot(231)
    imagesc(img); colormap(gray(256)); axis('equal'); axis('tight'); drawnow;
    subplot(232)
    imagesc(score);  colormap(gray(256)); axis('equal'); axis('tight'); ...
	title('Boosting output'); drawnow; 
    subplot(234)
    imagesc(img + 200*(score>0));  colormap(gray(256)); axis('equal'); ...
	axis('tight'); title('Boosting output'); drawnow; 
    subplot(235)
    imagesc(score>0);  colormap(gray(256)); axis('equal'); ...
	axis('tight'); title('Boosting output'); drawnow; 
    subplot(233)
    imagesc(mean(feature,3));  colormap(gray(256)); axis('equal'); ...
	axis('tight'); title('average of all features'); drawnow; 
    
    [y,x] = find(score == max(score(:))); x=x(1); y = y(1);
    subplot(232)
    hold on; plot(x,y,'ro'); hold off
    subplot(231)
    hold on; plot([x-16 x-16 x+16 x+16 x-16],[y-6 y+6 y+6 y-6 ...
		    y-6],'r','linewidth',3); hold off 

 %   figure, imshow(score, []);

end

end
