'''
File: dataset_handler.py
Author: Abdurahman Mohammed
Date: 2024-09-05
Description: A Python script that defines a PyTorch dataset class for the cell counting task.
'''

import torch
from torch.utils.data import Dataset

import pandas as pd
from PIL import Image


# A Pytorch dataset class
class CellDataset(Dataset):
    '''
        A PyTorch dataset class for the cell counting task. It takes a list of image paths as an argument and loads the images and their corresponding labels. The labels are the number of cells in the image. The images are loaded using the PIL library and transformed using the torchvision.transforms module. The transform argument is optional and can be used to apply data augmentation techniques to the images.
        
        Args:
            image_paths (list): A list that contains paths to the images.
            transform (torchvision.transforms): A transform object that will be applied to the images.

        Returns:
            img (torch.Tensor): A tensor that represents the image.
            label (torch.Tensor): A tensor that represents the label.

    '''
    def __init__(self, image_paths, transform=None):
        '''
            Initializes the dataset with a list of image paths and an optional transform object.

            Args:
                image_paths (list): A list that contains paths to the images.
                transform (torchvision.transforms): A transform object that will be applied to the images.
        '''
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Get the image path and the corresponding ground truth path
        img_path = self.image_paths[idx]
        gt_path = img_path.replace('images', 'ground_truth').replace('.tiff', '.csv')

        # Load the image and the ground truth
        img = Image.open(img_path).convert('RGB')
        gt = pd.read_csv(gt_path)
        label = torch.tensor(gt.shape[0], dtype=torch.float32).unsqueeze(0)

        # Apply the transform if available
        if self.transform:
            img = self.transform(img)

        return img, label
        
