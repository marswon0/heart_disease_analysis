# Heart Disease Recognition Through ML and DL

## Introduction

Heart disease can be hard to detect in the preliminary stages since the multiple health indicators must be considered. Machine 
learning has been proved to be effective in assisting decision making as well as classification. This project uses different machine learning and deep learning techniques to predict heart disease. Models used in this project include Support Vector Machine(SVM), Multilayer Perceptron(MLP), and Ensemble Learning methods. 

## Data Visualization

This section provides some vidualizations for the training samples. The proposed models are validated through the combined [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) (Cleveland, Hungary, Switzerland, and the VA Long Beach). The combined dataset has 1025 samples, each sample has 13 features.

<img src="/Assets/Images/visualization.JPG">

## Results

### SVM 

Different kernel functions have been tested for the SVM model. The performance of different kernel functions are summarized in the table below.

<img src="/Assets/Images/SVM.JPG" width="400" height="133">

### SVM with K-means

Performance of SVM with different numbers of clusters, all the simulations use the same SVM model with linear kernel. The accuracy of SVM is directly proportional to the number of clusters.

<img src="/Assets/Images/SVM_K_mean_linear_kernel.JPG" width="400" height="130">

Compared with standard SVM models, the classification accuracy can be further improved by integrating K-means to pre-process the training data.

<img src="/Assets/Images/SVM_K_mean_kernels.JPG" width="400" height="130">

### Multilayer Perceptron (MLP)

The MLP model reached 80% validation accuracy within 5 epochs using Adam optimizer. To slow down the learning process, a 30% drop-out is added after each of the dense layers. Different optimizers have been tested, as well as different loss functions.

- MLP training curve, SGD optimizer

<figure>
    <img src="/Assets/Images/MLP_SGD.JPG">
</figure>

- MLP training curve, Adam optimizer

<figure>
    <img src="/Assets/Images/MLP_Adam.JPG">
</figure>


Using the SGD optimizer, the model did not converge to an optimal solution, whereas the Adam optimizer could handle the gradients propagated in the MLP model. The final validation accuracy achieved by the MLP model is between 86~88%.

### Random Forest (RF)
    
In RF models, the maximum depth of each tree is set to 13 since the samples in the dataset contains 13 features. Either Gini or Entropy criteria can be selected for evaluating the quality of a split. Since the RF does not always generate the same result, multiple simulations have been done to validate the performance of Gini and Entropy criteria.
    
<img src="/Assets/Images/RF.JPG" width="550" height="200">    

### Ada Boosting

AdaBoost uses stump as the base estimator. The performance of AdaBoost is directly proportional to the amount of base estimator incorporated in the model. Compared with RF, AdaBoost had a more consistent performance. AdaBoost reached 90% validation accuracy through 100 base estimators. Although the validation accuracy reached 100% using 2000 estimators, the model is severely overfitting.

<img src="/Assets/Images/Ada.JPG" width="500" height="150">

## Usage

- To install the required packages, run ```pip install -r requirements.txt``` 
- To run a specific model, execute the jupyter notebook accordingly 

## Reference

For more information about this project, please check out this [paper](https://github.com/marswon0/heart_disease_recognition/blob/4c5babe3b714034d97a0a31cad2e4c826a31ded2/Assets/Paper/Heart%20Disease%20Recognition.pdf)
