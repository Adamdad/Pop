Derek Hoiem
Jan 03, 2006

***************************************************************
INTRODUCTION
***************************************************************

This document describes how to use the code provided for 
D. Hoiem, A.A. Efros, and M. Hebert, "Putting Objects in Perspective", CVPR 
2006.

This system has four components:

Geometry: 
Computes probability of each geometric class (ground, sky, vertical 
w/ subclasses) for each pixel in image.  Code provided in a separate package 
("Automatic Photo Pop-up" or APP).

Detection: 
Computes log-likelihood ratios of each bounding box containing each
object of interest in the image (standard window-based object detection).  
Code based on Murphy-Torralba-Freeman 2003, provided here.

Viewpoint: 
Hidden variables denoting the height and angle (given by horizon 
position) of camera.  Code computes priors from training data, provided here.

Inference: 
Compute marginals taking into account geometry, detection, and 
viewpoint marginals using belief propagation.  Code provided here.



***************************************************************
SUMMARY
***************************************************************

Training:
1) Estimate geometry priors
2) Learn object detectors
3) Estimate object priors
4) Estimate viewpoint priors

Testing:
1) Compute confidence images for geometric labels
2) Run object detection over images
3) Run inference function
4) Optionally, compute ROC curves for detections using candidates2det.m
   and detRoc.m


Other packages required:
* BNT for Matlab toolbox
* MATLAB Statistics Toolbox (if train/testing using MTF detector)
* PASCAL Grand Challenge Data (if training MTF detector)



***************************************************************
TRAINING
***************************************************************


Geometry: 
Classifiers are provided in the APP package.  

The geometry prior P(g) and
interaction likelihoods P(g|o) must be provided.  We include values for 
cars and people (along with prior) in pG_car_ppl.mat.  


Detection: 
Note that this detection code is based off code provided for the Murphy-
Torralba-Freeman 2003 classifier.  Other detectors can be used as well.

detTrainScript is the main file.  The imdir and datadir variables must be
set appropriately.  
detInitData requires the code provided by the PASCAL Object Recognition
Grand Challenge.  If training is not done using PASCAL data, a different 
function should be written to initialize the training data.  
We set USE_CONTEXT=0 and USE_COLOR=1 for our paper.

The object prior P(o) must also be estimated for each object.  We fit a 
sigmoid to the confidence estimates from a validation set using
getProbablisticOutputParams.  


Viewpoint:
Training involves estimating the prior for the horizon and camera height.
This involves manually labeling the horizon in a set of training images 
that have lableed objects, finding the ML estimate for the camera height 
(based on horizon and object height distributions), and estimating the 
pdfs.  The pdfs can be estimated using kernel density estimation or computing
the MLE for gaussian parameters.  We provide some tools that may make these
tasks easier:
labelhorizon.m - manually click on horizon positions in set of images
viewTrainCameraHeightPrior.m - get height prior from ML estimates





***************************************************************
TESTING
***************************************************************


Geometry:
Replace APPtestDirectory.m included in the APP code package with the one 
found in the geometry directory in this package.  Run this function to 
create the needed geometric context files.


Detection: 
detTestScript.m is the main script for computing detection window 
confidences for new images.  The imdir and outdir variables require 
modification.  


Viewpoint:
Nothing is needed for viewpoint, since only priors are used.


Inference:
Requires Matlab BNT package (available on sourceforge).  
Inference (computing of final estimates) can be performed using either
govInferenceScript.m or govInference.m.  
Note: The latest version (1.0.2) seems to have some problems with my code.  
May need to use version 1.0.1 instead.

govInferenceScript.m: 
Variables that point to the data locations will need 
to be adjusted.  Instead of using the ts structure that is loaded, the 
filenames can be obtained through other means.

govInference.m: 
Performs inference for one image given a set of object candidates, geometry
estimates, and prior parameters.

It may be useful to look at govInferenceScript.m to see how to acquire the 
needed inputs.  I should write a script around this. 


***************************************************************
NOTES
***************************************************************

HOW TO USE A DIFFERENT OBJECT DETECTOR:

Using an off-the-shelve object detector is easy.  There are two steps. 

The first step is to convert
the outputs of the detector to the "candidates" form.  A single candidate 
describes a set of bounding boxes, of which at most one can be a particular
object type.  Each element of candidates has an array of bounding boxes in 
the format [x1 x2 y1 y2], an array of confidences associated with each 
bounding box, and an object type number.  Look at detReadCandidatesScript and 
detReadAllCandidates for examples of how to make this conversion.  

Second, the confidences must be converted to probabilities, which can be done 
with a validation set  and the function getProbabilisticOutputParams.  I have 
used this to convert the log-likelihood ratios of Adaboost and SVM outputs to 
probabilities.  



