# Import necessary libraries
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    SaveImage
)

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
import os
import torch
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

# Define input and model directories
in_dir = '../'
model_dir = '../results/'

# Load training and testing loss and metric data
train_loss = np.load(os.path.join(model_dir, 'loss_train.npy'))
train_metric = np.load(os.path.join(model_dir, 'metric_train.npy'))
test_loss = np.load(os.path.join(model_dir, 'loss_test.npy'))
test_metric = np.load(os.path.join(model_dir, 'metric_test.npy'))

# Create a figure for plottings
plt.figure("Results", (12, 6))

# Plot training loss
plt.subplot(2, 2, 1)
plt.title("Train dice loss")
x = [i + 1 for i in range(len(train_loss))]
y = train_loss
plt.plot(x, y)

# Plot training metric
plt.subplot(2, 2, 2)
plt.title("Train metric dice")
x = [i + 1 for i in range(len(train_metric))]
y = train_metric
plt.plot(x, y)

# Plot testing loss
plt.subplot(2, 2, 3)
plt.title("Test dice loss")
x = [i + 1 for i in range(len(test_loss))]
y = test_loss
plt.plot(x, y)

# Plot testing metric
plt.subplot(2, 2, 4)
plt.title("Test metric dice")
x = [i + 1 for i in range(len(test_metric))]
y = test_metric
plt.plot(x, y)

# Save the figure as an image
plt.savefig("output/test.png")

# Get file paths for testing volumes and segmentations
path_test_volumes = sorted(glob(os.path.join(in_dir, "imagesVal", "*.nii.gz")))
path_test_segmentation = sorted(glob(os.path.join(in_dir, "labelsVal", "*.nii.gz")))

# Create a list of dictionaries specifying image-label pairs for testing
test_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(path_test_volumes, path_test_segmentation)]

# Define data transforms for testing
test_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5,1.5,1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True), 
        CropForegroundd(keys=['image', 'label'], source_key='image'),
        Resized(keys=["image", "label"], spatial_size=[128,128,64]),   
        ToTensord(keys=["image", "label"]),
    ]
)
test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1)

# Set the device for inference to CUDA if available, otherwise use CPU
device = torch.device("cuda:0")

# Create a U-Net model and load the best metric model checkpoint
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=17,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

model.load_state_dict(torch.load(os.path.join(model_dir, "best_metric_model.pth")))
model.eval()

# Perform inference on test data
with torch.no_grad():
    # Create a SaveImage object for saving output images
    saver = SaveImage(output_dir='output', output_postfix='test', output_ext=".nii.gz")
    for test_patient in test_loader:
        t_volume = test_patient['image']
        t_segmentation = test_patient['label']

        # Define sliding window inference parameters
        sw_batch_size = 4
        roi_size = (128, 128, 64)

        # Perform sliding window inference
        test_outputs = sliding_window_inference(t_volume.to(device), roi_size, sw_batch_size, model).argmax(dim=1)
        test_patient["output"] = test_outputs

        # Save the output images
        for data in decollate_batch(test_patient):
            saver(data["output"])
