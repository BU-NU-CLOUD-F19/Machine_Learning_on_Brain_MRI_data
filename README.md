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


### Out of Scope/Stretch goals:
*	Efficiency/accuracy of the trained model
*	Enhancement/Modification of chRis platform/architecture to run the application
*	Automating any of the scope goals in the platform
*	Any UI for the application


## 4. Solution Concept

### High Level Outline:

*	First major task is to understand the Brain MRI data that are in .mgz format and carefully pre-process to NiFTI/any ML recognizable format to create a standardize data for the ML models to be trained on. This task also includes preprocessing of  the labels to ML readable format for the training models.
The sample data visualization is shown below:

![Image description](https://github.com/BU-NU-CLOUD-F19/Machine_Learning_on_Brain_MRI_data/blob/master/PACSPull_Output.png)


### The two major plugin description is as mentioned below.


*	Create a plug-in using ChRIS cookie cutter module to develop a ML model using python to take these pre-processed data and labels and train the model so that it is able to classify on test dataset. Save these trained models to an output location so that it can used by the next layer/plug-in.
*	Create a plug-in using ChRIS cookie cutter to develop an inference layer using python that will take the saved train models from the above layer and classify any unseen brain MRI data and save the inference in an output location.The overall flow diagram for both of these application is shown below.

![Image description](https://github.com/BU-NU-CLOUD-F19/Machine_Learning_on_Brain_MRI_data/blob/master/Screen%20Shot%202019-09-24%20at%203.15.16%20PM.png)



The flow of data in the whole system will be from d0 which is input data to our first application and contains brain MRI images as well labels to train a machine learning model that segments these images and generates a model file(.pb file) as an output o0. This pretrained model will be used by the second application which does the inference on test images and generate segmentation for these images as an output.

### Machine learning methodology

* We are going to use tensorflow as a main tool to create a machine learning model and train on our data. The machine learning workflow is explained in the image below.









## 5. Acceptance criteria of the project

*	Correctly pre-process the Brain MRI images for the training model to be trained on
*	The training model should be able to fetch pre-processed data from an input location and store a trained model on some output location
*	The inference model should be able to pick up the trained models and classify any new images and store it in some output location
*	All these layers should be integrated in the cloud platform so that it can generate a docker image to be used on a linux kernel



## 6. Release Planning 

### Release #1

* Test the existing docker image that performs classification on MNIST dataset on local environment
* Collect and preprocess training data

### Release #2

* Design a basic network capable of training the given dataset
* Train the neural network using new data
