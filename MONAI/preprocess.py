# Import necessary libraries
import os
from glob import glob
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
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism

def prepare(in_dir, pixdim=(1.5, 1.5, 2.0), a_min=-200, a_max=200, spatial_size=[128,128,64], cache=False):
    '''
    Prepares and returns data loaders for training and testing.

    Args:
        in_dir (str): Directory containing input data.
        pixdim (tuple, optional): Voxel dimensions (default: (1.5, 1.5, 2.0)).
        a_min (float, optional): Minimum intensity for scaling (default: -200).
        a_max (float, optional): Maximum intensity for scaling (default: 200).
        spatial_size (list, optional): Spatial size for resizing (default: [128, 128, 64]).
        cache (bool, optional): Whether to use cache for data loading (default: False).

    Returns:
        tuple: (train_loader, test_loader) - Data loaders for training and testing.
    '''
    
    # Set determinism for reproducibility
    set_determinism(seed=0)

    # Get file paths for training and testing images and labels
    path_train_images = sorted(glob(os.path.join(in_dir, 'imagesTr', "*.nii.gz")))
    path_train_labels = sorted(glob(os.path.join(in_dir, 'labelsTr', '*.nii.gz')))
    path_val_images = sorted(glob(os.path.join(in_dir, 'imagesVal', '*.nii.gz')))
    path_val_labels = sorted(glob(os.path.join(in_dir, 'labelsVal', '*.nii.gz')))

    # Create a list of dictionaries specifying image-label pairs for training and testing  
    train_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(path_train_images, path_train_labels)]
    test_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(path_val_images, path_val_labels)]

    # Define data transforms for training and testing
    train_transforms = Compose(
        [
            # Load image and label data from keys "image" and "label"
            LoadImaged(keys=["image", "label"]),
            # Add a channel dimension to the image and label tensors
            AddChanneld(keys=["image", "label"]),
            # Rescale voxel spacings to (1.5, 1.5, 1.0) with bilinear interpolation for image, and nearest interpolation for label
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            # Adjust the orientation of the image and label to anatomical standard "RAS"
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Scale image intensity values from the range [-200, 200] to [0.0, 1.0]
            ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), 
            # Crop the foreground region of the image and label based on the "image" key
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # Resize both the image and label to a spatial size of [128, 128, 64]
            Resized(keys=["image", "label"], spatial_size=spatial_size),   
            # Convert image and label tensors to PyTorch tensors
            ToTensord(keys=["image", "label"]),
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max,b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            Resized(keys=["image", "label"], spatial_size=spatial_size),   
            ToTensord(keys=["image", "label"]),
        ]
    )

    if cache:
        # Create cache-based datasets and data loaders
        train_ds = CacheDataset(data=train_files, transform=train_transforms,cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader

    else:
        # Create non-cache-based datasets and data loaders
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader
