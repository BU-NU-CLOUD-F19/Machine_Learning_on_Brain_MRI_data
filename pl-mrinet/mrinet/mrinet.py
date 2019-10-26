#!/usr/bin/env python
#
# mrinet ds ChRIS plugin app
#
# (c) 2016-2019 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#

from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras.models as models
from skimage.transform import resize
from skimage.io import imsave
import numpy as np

np.random.seed(256)
import tensorflow as tf

from keras.models import Model,load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling3D, ZeroPadding3D
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model
import os
K.set_image_data_format('channels_last')
import cv2
import sys
sys.path.append(os.path.dirname(__file__))

# import the Chris app superclass
from chrisapp.base import ChrisApp


Gstr_title = """

Generate a title from
http://patorjk.com/software/taag/#p=display&f=Doom&t=mrinet

"""

Gstr_synopsis = """

(Edit this in-line help for app specifics. At a minimum, the
flags below are supported -- in the case of DS apps, both
positional arguments <inputDir> and <outputDir>; for FS apps
only <outputDir> -- and similarly for <in> <out> directories
where necessary.)

    NAME

       mrinet.py

    SYNOPSIS

        python mrinet.py                                         \\
            [-h] [--help]                                               \\
            [--json]                                                    \\
            [--man]                                                     \\
            [--meta]                                                    \\
            [--savejson <DIR>]                                          \\
            [-v <level>] [--verbosity <level>]                          \\
            [--version]                                                 \\
            <inputDir>                                                  \\
            <outputDir>

    BRIEF EXAMPLE

        * Bare bones execution

            mkdir in out && chmod 777 out
            python mrinet.py   \\
                                in    out

    DESCRIPTION

        `mrinet.py` ...

    ARGS

        [-h] [--help]
        If specified, show help message and exit.

        [--json]
        If specified, show json representation of app and exit.

        [--man]
        If specified, print (this) man page and exit.

        [--meta]
        If specified, print plugin meta data and exit.

        [--savejson <DIR>]
        If specified, save json representation file to DIR and exit.

        [-v <level>] [--verbosity <level>]
        Verbosity level for app. Not used currently.

        [--version]
        If specified, print version number and exit.

        [--mode <mode>]
        Should be specified,
        If the <mode> is train, model will be trained.
        If the <mode> is infer, test images will be inferred from model.

"""


class Mrinet(ChrisApp):
    """
    An app to ....
    """
    AUTHORS                 = 'FNNDSC (dev@babyMRI.org)'
    SELFPATH                = os.path.dirname(os.path.abspath(__file__))
    SELFEXEC                = os.path.basename(__file__)
    EXECSHELL               = 'python3'
    TITLE                   = 'A ChRIS plugin app'
    CATEGORY                = ''
    TYPE                    = 'ds'
    DESCRIPTION             = 'An app to ...'
    DOCUMENTATION           = 'http://wiki'
    VERSION                 = '0.1'
    ICON                    = '' # url of an icon image
    LICENSE                 = 'Opensource (MIT)'
    MAX_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MIN_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MAX_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MIN_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MAX_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_GPU_LIMIT           = 0  # Override with the minimum number of GPUs, as an integer, for your plugin
    MAX_GPU_LIMIT           = 0  # Override with the maximum number of GPUs, as an integer, for your plugin

    # Use this dictionary structure to provide key-value output descriptive information
    # that may be useful for the next downstream plugin. For example:
    #
    # {
    #   "finalOutputFile":  "final/file.out",
    #   "viewer":           "genericTextViewer",
    # }
    #
    # The above dictionary is saved when plugin is called with a ``--saveoutputmeta``
    # flag. Note also that all file paths are relative to the system specified
    # output directory.
    OUTPUT_META_DICT = {}

    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        Use self.add_argument to specify a new app argument.
        """
        self.add_argument('--mode', dest='mode', type=str, optional=False,
                          help='What do you want to do 1. Train 2. Infer')
    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


    def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)


    def get_unet(self):
        inputs = Input((256, 256, 1))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)


        model = Model(inputs=[inputs], outputs=[conv10])

        model.summary()
        #plot_model(model, to_file='model.png')

        model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss='binary_crossentropy', metrics=['accuracy'])

        return model



    def get_train_data(self,options):
        train_data = np.ndarray((256,256,256),dtype=np.uint8)
        in_files = os.listdir(options.inputdir + "/input_images")
        for i in in_files:
            np.append(train_data,cv2.imread(options.inputdir + "/" + i))
        return train_data

    def get_label_data(self,options):
        label_data = np.ndarray((256,256,256),dtype=np.uint8)
        label_files = os.listdir(options.inputdir +"/label_images")
        for i in label_files:
            np.append(label_data,cv2.imread(options.inputdir + "/" + i))
        return label_data

    def get_test_data(self,options):
        test_data = np.ndarray((1,256,256),dtype=np.uint8)
        test_files = os.listdir(options.inputdir +"/test_images")
        for i in test_files:
            np.append(test_data,cv2.imread(options.inputdir + "/" + i))
        return test_data


    def run(self, options):
        if options.mode == "1":
            self.train(options)
        elif options.mode == "2":
            self.predict(options)
        else:
            print("You have selected invalid option for conversion")

    def train(self,options):
            model = self.get_unet()
            train_data = self.get_train_data(options)
            label_data = self.get_label_data(options)
            train_data = np.expand_dims(train_data,axis=3)
            label_data = np.expand_dims(label_data,axis=3)          
            model.fit(train_data,label_data,epochs=1,batch_size=1,verbose=1,shuffle=True,validation_split=0.8)
            model.save(options.outputdir + "/model.h5")
            print("in train method")

    def predict(self,options):
            model = self.get_unet()
            model = load_model(options.outputdir + "/model.h5")
            test_data = self.get_test_data(options)
            test_data = np.expand_dims(test_data,axis=3)
            cv2.imwrite(options.outputdir + "/inference_image.jpg",model.predict(test_data))
            print("in predict method")

    def show_man_page(self):
            """
            Print the app's man page.
            """
            print(Gstr_synopsis)


        


# ENTRYPOINT
if __name__ == "__main__":
    chris_app = Mrinet()
    chris_app.launch()
