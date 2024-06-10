import nibabel as nib
import os
from scipy.ndimage import zoom
import numpy as np
from torch import argmax

class Data_Reader():
    def __init__(self, file_name, image_path, mask_path=None):
        self.image = None
        self.mask = None
        self.x_dim = None
        self.y_dim = None
        self.z_dim = None
        self.org_dim = None
        self.top_pad = None
        self.bottom_pad = None
        self.nn_output = None

        if image_path is not None:
            self.image = nib.load(os.path.join(image_path, file_name))
        if mask_path is not None:
            self.mask = nib.load(os.path.join(mask_path, file_name))
        
        self.compute_z_dim()
    
    def process_image(self):
        # get data
        transform = self.image.get_fdata()

        # scale and downsample
        transform = self.scale(transform)
        
        # padding
        self.compute_pad(transform)
        transform = self.pad(transform, -1000)

        # normalize
        transform = self.normalize(transform)

        return transform.astype(np.float32)
    
    def process_mask(self):

        transform = self.mask.get_fdata()

        transform = self.scale(transform, 0)

        transform = self.pad(transform, 0)

        return transform.astype(np.int64)

    def compute_z_dim(self):
        self.z_dim = self.image.header.get_zooms()[2]
        self.y_dim = self.image.header.get_zooms()[1]
        self.x_dim = self.image.header.get_zooms()[0]
    
    def scale(self, data, order=3):
        return zoom(data, ((0.5, 0.5, self.z_dim / 2)), order=order)
    
    def compute_pad(self, data):
        pad_needed = 576 - data.shape[2]
        self.bottom_pad = int(pad_needed / 2)
        self.top_pad = pad_needed - self.bottom_pad

    def pad(self, data, value):
        return np.pad(data, ((0, 0), (0, 0), (self.top_pad, self.bottom_pad)), mode="constant", constant_values=value)
    
    def normalize(self, data):
        output = (data + 1000.0) / 5000.0
        return output
    
    def set_pred_mask(self, mask):
        self.nn_output = mask.numpy()
        self.nn_output = np.squeeze(self.nn_output)

    def process_pred_mask(self):
        self.un_pad()
        self.upscale_mask()

    def un_pad(self):
        self.nn_output = self.nn_output[ : , : , self.top_pad: -1 * self.bottom_pad]
    
    def upscale_mask(self):
        self.nn_output = zoom(self.nn_output, (2, 2, 2.0 / self.z_dim), order=0)
    
    def get_pred_mask(self):
        return self.nn_output.astype(np.uint8)
    
    def create_nifti_mask(self, file_name):
        affine = self.image.affine
        header = self.image.header
        header.set_data_dtype(np.uint8)
        image = nib.nifti1.Nifti1Image(self.get_pred_mask(), affine, header)
        nib.save(image, file_name)
