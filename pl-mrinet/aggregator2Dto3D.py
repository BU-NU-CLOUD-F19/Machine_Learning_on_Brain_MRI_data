import os
from skimage.io import imread
import nibabel as nib
import numpy as np
import nibabel.freesurfer.mghformat as mgh
png_files=os.listdir("mask")

def preprocess(imgs):
    imgs = np.expand_dims(imgs, axis=3)
    return imgs

def preprocess_squeeze(imgs):
    imgs = np.squeeze(imgs, axis=3)
    return imgs


number_of_files=len(png_files)
image_rows = int(256)
image_cols = int(256)
image_depth = 5
print(' ---------------- image reconstruction in progress -----------------')
imgs_temp = np.ndarray((number_of_files, image_rows, image_cols), dtype=np.uint8)
for i in range(0,number_of_files):
    img=((imread("mask/"+png_files[i])))
    img = img.astype(np.uint8)
    img = np.array([img])
    imgs_temp[i] = img
imgs_temp = np.around(imgs_temp, decimals=0)
imgs_temp = (imgs_temp * 255).astype(np.uint8)

affine = np.diag([1, 2, 2, 1])
array_img=nib.Nifti1Image(imgs_temp,affine)
print(' ---------------- image reconstructed -----------------')
array_header = array_img.header
mgh.save(scaled_img,"img.mgz")
print(' ---------------- image saved to local drive -----------------')

