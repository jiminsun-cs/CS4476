"""
Contains functions with different data transforms
"""

from typing import Tuple

import torchvision.transforms as transforms
import numpy as np


def get_fundamental_transforms(
    inp_size: Tuple[int, int], pixel_mean: np.array, pixel_std: np.array
) -> transforms.Compose:
    """
    Returns the core transforms needed to feed the images to our model (refer notebook for the 4 operations).

    Args:
    - inp_size: tuple denoting the dimensions for input to the model
    - pixel_mean: the mean of the raw dataset [Shape=(1,)]
    - pixel_std: the standard deviation of the raw dataset [Shape=(1,)]
    Returns:
    - fundamental_transforms: transforms.Compose with the 4 fundamental transforms
    """

    #comment the raise error
    # raise NotImplementedError('get_fundamental_transforms not implemented')

    return transforms.Compose(
        [
            ############################################################################
            # Student code begin
            ############################################################################
            transforms.Resize(inp_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(pixel_mean, pixel_std)
            ############################################################################
            # Student code end
            ############################################################################
        ]
    )


def get_data_augmentation_transforms(
    inp_size: Tuple[int, int], pixel_mean: np.array, pixel_std: np.array
) -> transforms.Compose:
    """
    Returns the data augmentation + core transforms needed to be applied on the train set. Put data augmentation transforms before code transforms. 

    Note: You can use transforms directly from torchvision.transforms

    Suggestions: Jittering, Flipping, Cropping, Rotating.

    Args:
    - inp_size: tuple denoting the dimensions for input to the model
    - pixel_mean: the mean of the raw dataset
    - pixel_std: the standard deviation of the raw dataset
    Returns:
    - aug_transforms: tra nsforms.compose with all the transforms
    """
    # raise NotImplementedError('get_data_augmentation_transforms not implemented')
    
    return transforms.Compose(
        [
            ############################################################################
            # Student code begin
            ############################################################################
            transforms.Resize(inp_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3, saturation=0.3),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(),
            transforms.Normalize(pixel_mean, pixel_std)           
        
            ############################################################################
            # Student code end
            ############################################################################
        ]
    )
