# GLVQ_Renyi_Divergence
The implementation of statistical-based distance measure (Renyi-Alpha Divergence) in Generalized Vector Quantization (GLVQ) algorithm. GLVQ is a popular classification algorithm.

##Pseudo-Code

* Get labeled data.
* Intialize prototypes with labels.
* Implement L1 normalization in both input data and prototypes.
* Calculate Renyi-Alpha Divergence (distance) between datapoints and prototypes.
* Get the minimum distance from datapoints to each prototype with the same label by using Renyi alpha distance measure.
* Get the minimum distance from datapoints to each prototype with different label by using Renyi alpha distance measure.
* Calculate cost function.
* Finally, update the prototypes.
* Measure the success of the classification task using metrics such as Confusion matrix, Precision, Recall, F1 score and ROC curve.

##Installation

* Clone this repository
* Make sure that _numpy_, _matplotlib_, _Scikit Learn_ are installed in your environment.
* Go to folder and run the 'Renyi_final' file.



