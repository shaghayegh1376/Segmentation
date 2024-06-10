import numpy as np
import h5py
import nibabel as nib
import os
from scipy.ndimage import zoom

class File():
    def __init__(self, image_dir, labels_dir, file_name):
        self.file_name = file_name
        image = nib.load(os.path.join(image_dir, file_name))
        label = nib.load(os.path.join(labels_dir, file_name)).get_fdata().astype(np.int64)
        self.z_dim = image.header.get_zooms()[2]
        image = image.get_fdata()

        self.image = zoom(image, (0.5, 0.5, self.z_dim / 2))
        self.label = zoom(label, (0.5, 0.5, self.z_dim / 2), order=0)

    def pad(self):
        pad_needed = 576 - self.image.shape[2]
        bottom_pad = int(pad_needed / 2)
        top_pad = pad_needed - bottom_pad
        self.image = np.pad(self.image, ((0, 0), (0, 0), (top_pad, bottom_pad)), mode="constant", constant_values=-1000.0)
        self.label = np.pad(self.label, ((0, 0), (0, 0), (top_pad, bottom_pad)), mode="constant", constant_values=0)

    def normalize(self):
        self.image = (self.image + 1000.0) / 5000.00

    def get_image(self) -> np.ndarray:
        return self.image.astype(np.float32)
    
    def get_label(self) -> np.ndarray:
        assert(self.label.max() <= 16)
        assert(self.label.min() >= 0)
        return self.label.astype(np.int64)

def verify(images, file_name):
    issues = list()
    h5f = h5py.File(file_name, 'r')
    for image in images:
        try:
            data = h5f[image][:]
        except:
            issues.append(image)
    h5f.close()
    return issues

if __name__ == "__main__":
    files = os.listdir(os.path.join("WORD-V0.1.0", "imagesTr"))
    h5f = h5py.File("WORD_Tr_img.h5", 'w')
    h5b = h5py.File("WORD_Tr_mask.h5", 'w')

    for file in files:
        print(file, " Opening")
        image = File(os.path.join("WORD-V0.1.0", "imagesTr"), os.path.join("WORD-V0.1.0", "labelsTr"), file)
        image.pad()
        image.normalize()
        
        h5f.create_dataset(file, data=image.get_image())
        h5b.create_dataset(file, data=image.get_label())
        print(file, " Done!")
    
    h5f.close()
    h5b.close()

    issues = verify(files, "WORD_Tr_img.h5")
    print(f"Image files with issues: {issues}")

    issues = verify(files, "WORD_Tr_mask.h5")
    print(f"Mask files with issues: {issues}")

    