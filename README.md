# Machine_Learning_on_Brain_MRI_data

## 1. Vision and goals of the project

We will develop two containarized applications that run on ChRIS cloud computing platform as well as work as standalone applications to help segment brain MRI data for researchers.

High level goals include

* A containerized application that trains convolutional network using precomputed features to optimize a model to perform brain region segmentation

* A containerized application that will use this pretrained model to segment new images


## 2. Users/Personas Of The Project

### End Users

* Clinical Researchers


## 3. Scope and Features

* The applications should be able to run as a ChRIS plugin or a standalone application on

## 4. Solution Concept

### High Level Outline:

<img src="https://github.com/BU-NU-CLOUD-F19/Machine_Learning_on_Brain_MRI_data/blob/master/Screen%20Shot%202019-09-16%20at%201.22.50%20PM.png" height="200" />

Source-to-Image (S2I) is a framework that makes it easy to write images that take application source code as an input and produce a new image that runs the assembled application as output.

We will use S2I to package our code along with all the dependencies to create an image which will run on docker.



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
