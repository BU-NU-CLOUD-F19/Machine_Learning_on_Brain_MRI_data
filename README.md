# Machine Learning on Brain MRI data

## 1. Vision and goals of the project

#### Vision 

* Expand an existing Cloud based Machine Learning application that is trained on MNIST data to identify digits, to classify Brain MRI data and make the application compatible with ChRIS, an opensource distributed software platform(https://github.com/FNNDSC/CHRIS_docs).


#### High level goals include:
*	Pre-process the Brain MRI data that are in. mgz form to NIfTI(https://nifti.nimh.nih.gov) or .png format so that it will be easy for ML models to understand
*	Create a ChRIS plugin to do the pre-processing of the input images.
*	Create a ChRIS plugin to train an ML model on the pre-processed MRI data and save the trained model in an output location.
*	Create a ChRIS plugin to infer from the saved trained model and store the classified images in an output location.



## 2. Users/Personas Of The Project

* The application or plugins will be used by clinicians and researchers who wants to segment the Brain MRI data which is in .mgz 3D image format to analyze the MRI in a better way. Also Machine learning researchers who wants to try out different models without worrying about the environment and data preprocessing setup can use this application.


## 3. Scope and Features

### Major Features include

*	To create major plugins: 
1. A plugin to preprocess the input .mgz images. This plugin should read .mgz images that are uploaded for inference or training from given input location and should preprocess them to a machine learning compatible format and store it in an output folder.
2. A plugin to train the machine learning model with images present in an input directory and save the trained model in an output directory, where the input and output directories are given as a positional arguments in the docker run command.
3. A plugin which loads the saved model and test images from input directories and saves the inferred segmented images to an output directory, where the input and output directories are given as a positional arguments in the docker run command.

### Out of Scope/Stretch goals:
*	Efficiency/accuracy of the trained model
*	Enhancement/Modification of chRis platform/architecture to run the application
*	Automating any of the scope goals in the platform
*	Any UI for the application


## 4. Solution Concept

### High Level Outline:

*	First major task is to understand the Brain MRI data that are in .mgz format and carefully pre-process to NiFTI/any ML recognizable format to create a standardize data for the ML models to be trained on. This task also includes preprocessing of  the labels to ML readable format for the training models.
The sample data visualization is shown below:

![Image description](https://github.com/BU-NU-CLOUD-F19/Machine_Learning_on_Brain_MRI_data/blob/master/images/PACSPull_Output.png)


### The two major plugin description is as mentioned below.


*	Create a plug-in using ChRIS cookie cutter module to develop a ML model using python to take these pre-processed data and labels and train the model so that it is able to classify on test dataset. Save these trained models to an output location so that it can used by the next layer/plug-in.
*	Create a plug-in using ChRIS cookie cutter to develop an inference layer using python that will take the saved train models from the above layer and classify any unseen brain MRI data and save the inference in an output location.The overall flow diagram for both of these application is shown below.

![Image description](https://github.com/BU-NU-CLOUD-F19/Machine_Learning_on_Brain_MRI_data/blob/master/images/Screen%20Shot%202019-09-24%20at%203.15.16%20PM.png)



The flow of data in the whole system will be from d0 which is input data to our first application and contains brain MRI images as well labels to train a machine learning model that segments these images and generates a model file(.pb file) as an output o0. This pretrained model will be used by the second application which does the inference on test images and generate segmentation for these images as an output.

The above plugin's should be compatible with the ChRIS computing platform. To ensure this compatibilty we need to make sure that the application is containerized with docker and the application takes two positional arguments to serve as input and output directories, from where the input to application can be read from and the output of the application will be written to.

The sinppet below shows a ChRIS plugin being run using the docker run command with two positional arguments for input and output directories.

```
docker run -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing       \
        fnndsc/pl-simpledsapp simpledsapp.py                    \
        --prefix test-                                          \
        --sleepLength 0                                         \
        /incoming /outgoing
```


### Machine learning methodology

* We are going to use tensorflow as a main tool to create a machine learning model and train on our data. The main machine learning challenge is to develop a neural network architecture that can take as input 3D images and generate same size of masks as an output. This problem is known as 3D image segmentation.


![](https://github.com/BU-NU-CLOUD-F19/Machine_Learning_on_Brain_MRI_data/blob/master/images/Screen%20Shot%202019-09-27%20at%209.11.37%20AM.png)









## 5. Acceptance criteria of the project

*	Correctly pre-process the Brain MRI images for the training model. That is convert the image from .mgz format to ML understandable format.
*	The training model should be able to fetch pre-processed data from an input directory and store the trained model in an output directory.
*	The inference model should be able to pick up the trained model and classify any new images and store it in an output directory



## 6. Release Planning 

### Sprint #1

This sprint will mainly focus on setting up the infrastructure and understanding of the plugins that already exist. For this sprint major tasks are as described below.
* Setting up environment for and run existing application that trains a neural network to classify on MNIST datset.
* Setting up a sample ChRIS plugin using Cookie Cutter 
* Transfer brain data and organize inputs and labels for training
* Finding a method to visualize and load this .mgz files in python
* Survey existing literature for 3D image segmentation

#### Sprint #1 demo presentation: https://docs.google.com/presentation/d/1F1Ue_Y8czt1cEbp_1YD7r1CQM0LiCNneU5yJLxmNaBk/edit?usp=sharing


### Sprint #2

For this sprint we will focus on converting an existing application that trains neural network inside a docker image and uses s2i to a ChRIS compatible application and start working on a plugin that can preprocess .mgz images and convert them to a ML readable format. The major tasks include:

* Make the existing prototype application compatible with ChRIS.
* Write a script to create Pre processing data into matrix format.
* Create a simple plugin similar to a simpledsapp(https://github.com/FNNDSC/pl-simpledsapp) for preprocessing of images
* Model a simple neural data that can use preprocessed data as input

### Sprint #2 demo presentation: https://docs.google.com/presentation/d/10toYJOk46RZsSeF8bmHmCZ0BbfwQ5Ig_u7Uicdi5GcE/edit?usp=sharing


### Sprint #3

In the previous sprint we have already finished creating ChRIS plugin to convert mgz files to jpeg or numpy arrays as we are going to try 2D and 3D UNET models. This sprint we will work towards training plugin.

* Write a neural network that can segment 2D images according to the paper finalised.
* Write a neural network that can segment 3D images according to the paper.
* Create a ChRIS compatible application to train and infer from given dataset.

### Sprint #3 Demo Presentation : 
https://docs.google.com/presentation/d/1VsX2K0JT3BNESlGc5ob16MMkdHf2oEWAf01OMnuB8nM/edit?usp=sharing

### Sprint #4
* Setup Swift and push mgz data to swift.
* Creating an Aggregator to reconstruct a 3D image.
* Feed the input data using FS plugin.
* Run the training and inference plugin using S2I.

### Sprint #4 Demo Presentation:
https://docs.google.com/presentation/d/1XprRZ43IxtRajFhGqwB32l9Gx5X1x1KQM6tadN1fsPs/edit?usp=sharing

### Sprint #5
* Chaining all the plugins and running on ChRIS
* Spike: Investigate if UI can be used to connect the plugins and upload data to Swift

### Sprint #5 Demo Presentation
https://docs.google.com/presentation/d/15b81vt4P4kk22WpDI7vT-K6lNURinmsTu-yRgD-IGIQ/edit?usp=sharing


### Final Presentation Video
https://youtu.be/rFNroX6WVT0

### Steps to install the plugins locally
To install the plugin which converts the data from .mgz to .png follow the steps mentioned in readme which is present in ``pl-mgz_converter`` directory. Found here: pl-mgz_converter/README.rst

To install the plugin which trains the model and generateed segmented images follow the steps mentioned in readme which is present in ``pl-mrinet`` directory. Found here: pl-mrinet/README.rst
