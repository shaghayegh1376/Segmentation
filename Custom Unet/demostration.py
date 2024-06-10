import load_from_nifti_v2 as load
import numpy as np
import torch
import sys

# python demonstration.py filename, imagepath, output name (include the .nii.gz)

device = "cuda:2"

dataRead = load.Data_Reader(sys.argv[1], sys.argv[2])
image = dataRead.process_image()
image = np.expand_dims(image, axis=0)
image = np.expand_dims(image, axis=0)
image = torch.from_numpy(image)

image = image.to(device)

model = torch.load("trained_3DUnetDiceCE.pth")

model = model.to(device)
model.eval()

with torch.no_grad():
    output = model(image)

output = output.to("cpu")
output = torch.argmax(output, dim=1)

dataRead.set_pred_mask(output)
dataRead.process_pred_mask()
dataRead.create_nifti_mask(sys.argv[3])


