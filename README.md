# Machine Learning on Brain MRI data

## 1. Vision and goals of the project

#### Vision 

* The main idea of the project is to expand an existing Cloud application that is trained on MNIST data to identify digits, to classify Brain MRI data. The cloud platform we will use for this project is ChRIS which is a collaboration between Boston Children's Hospital and Redhat.


#### High level goals include:
*	Pre-process the Brain MRI data that are in. mgz form to NIfTI(https://nifti.nimh.nih.gov) format so that it will be easy for ML models to understand
*	Create a ChRIS plugin to train an ML model on the pre-processed MRI data and save the trained model in an output location
*	Create a ChRIS plugin to infer from the saved trained model and store the classified images in an output location
*	Package the application so that it can run on any linux based kernel



## 2. Users/Personas Of The Project

* The application or plugins will be used by clinicians and researchers


## 3. Scope and Features

### Major Features include

*	To create 2 plugins: Training Layer & Inference Layer in the existing application to use ML to classify Brain MRI data
* Modify the existing application to train on new Brain MRI data of .mgz type

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
