# Import necessary libraries
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
import torch
from preprocess import prepare
from utilities import train

# Define the directory where model-related files will be saved
model_dir = '../results' 

# Prepare the data using the 'prepare' function from the 'preprocess' module
data_in = prepare('../', cache=True)

# Set the device for training to CUDA if available, otherwise use CPU
device = torch.device("cuda:0")

# Define the U-Net model architecture
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=17,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

# Define the loss function (Dice loss)
loss_function = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True)

# Define the optimizer (Adam optimizer) for model parameter optimization
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

if __name__ == '__main__':
    # Train the model using the 'train' function, specifying the model, data, loss function, optimizer, and other settings
    train(model, data_in, loss_function, optimizer, 600, model_dir)
