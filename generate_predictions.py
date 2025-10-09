'''
File: generate_predictions.py
Author: Abdurahman Mohammed
Date: 2024-09-05
Description: A Python script that generates predictions using a trained model and saves them to a CSV file.

'''


import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from glob import glob
from tqdm import tqdm
from dataset_handler import CellDataset
from model import CellCounter
import pandas as pd


def load_model(checkpoint_path):
    '''
        Loads the model from a checkpoint file.

        Args:
            checkpoint_path (str): The path to the checkpoint file.

        Returns:
            model (CellCounter): The model loaded from the checkpoint file.
    '''
    model = CellCounter()
    model.load_state_dict(torch.load(checkpoint_path))
    return model


def get_data_loader(image_paths, batch_size=8):
    '''
        Creates a PyTorch DataLoader object for the given image paths.

        Args:
            image_paths (list): A list of image paths.
            batch_size (int): The batch size for the DataLoader.

        Returns:
            loader (DataLoader): A DataLoader object for the given image paths.
    '''

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = CellDataset(image_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader


def generate_predictions(model, data_loader, output_file):
    '''
        Generates predictions using the given model and DataLoader and saves them to a CSV file.

        Args:
            model (CellCounter): The model to use for generating predictions.
            data_loader (DataLoader): The DataLoader object to use for loading images.
            output_file (str): The path to the output CSV file.

        Returns:
            df (pd.DataFrame): A pandas DataFrame containing the image paths and predictions.

    '''


    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    predictions = []
    for inputs, _ in tqdm(data_loader, desc="Generating predictions"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.extend(outputs.squeeze().tolist())
    
    image_paths = data_loader.dataset.image_paths
    df = pd.DataFrame({"image_path": image_paths, "predictions": predictions})
    df.to_csv(output_file, index=False)
    return df


def main(checkpoint_path, image_dir, output_file):
    # Load a model from a checkpoint file
    model = load_model(checkpoint_path)
    
    # Get a list of image paths
    image_paths = glob(f"{image_dir}/*.tiff")
    
    # Create a DataLoader object
    data_loader = get_data_loader(image_paths)
    
    # Generate predictions and save them to a CSV file
    df = generate_predictions(model, data_loader, output_file)
    
    print(f"Predictions saved to {output_file}")
    
    return df


if __name__ == "__main__":
    main("model.pth", "IDCIA/test/images", "predictions.csv")
