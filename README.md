# Machine_Learning_on_Brain_MRI_data

## 1. Vision and goals of the project

We will develop two containarized applications that run on ChRIS cloud computing platform as well as work as standalone applications to help segment brain MRI data for researchers.

High level goals include

* A containerized application that trains convolutional network using precomputed features to optimize a model to perform brain region segmentation

* A containerized application that will use this pretrained model to segment new images


## 2. Users/Personas Of The Project

### End Users

* Clinical Researchers


## 5. Acceptance criteria of the project

* Use the first application to train neural network on data provided.

* Perform test on final classifier on new data provided by Boston Children's hospital in the MOC.

## 6. Release Planning 

### Release #1

* Test the existing docker image that performs classification on MNIST dataset on local environment
* Collect and preprocess training data

### Release #2

* Design a basic network capable of training the given dataset
* Train the neural network using new data
