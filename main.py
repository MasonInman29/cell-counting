'''
File: main.py
Author: Abdurahman Mohammed
Date: 2024-09-05
Description: A Python script that trains a cell counting model using the IDCIA dataset.

'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset_handler import CellDataset
from model import CellCounter

def get_data_loaders(batch_size=8):
    '''
    Creates training and validation data loaders for the IDCIA dataset.

    Args:
        batch_size (int): The batch size for the data loaders.

    Returns:
        train_loader (DataLoader): A DataLoader object for the training set.
        val_loader (DataLoader): A DataLoader object for the validation set.
    '''

    # Get training and testing image paths
    train_image_paths = glob("IDCIA/train/images/*.tiff")
    val_image_paths = glob("IDCIA/val/images/*.tiff")

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Create datasets
    train_dataset = CellDataset(train_image_paths, transform=train_transform)
    val_dataset = CellDataset(val_image_paths, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3):
    '''
    Trains the cell counting model on the training set and evaluates it on the validation set.

    Args:
        model (CellCounter): The model to train.
        train_loader (DataLoader): The DataLoader for the training set.
        val_loader (DataLoader): The DataLoader for the validation set.
        num_epochs (int): The number of epochs to train the model.
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        model (CellCounter): The trained model.
        train_losses (list): A list of training losses for each epoch.
        val_losses (list): A list of validation losses for each epoch.

    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss:.4f}")

        # Evaluate the model on the validation set
        val_loss = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")


    return model, train_losses, val_losses

def evaluate_model(model, data_loader, criterion, device):
    '''
        Evaluates the model on a validation or test set.

        Args:
            model (CellCounter): The model to evaluate.
            data_loader (DataLoader): The DataLoader for the validation or test set.
            criterion (torch.nn.Module): The loss function to use.
            device (torch.device): The device to run the evaluation on.

        Returns:
            total_loss (float): The average loss over the dataset.

    '''
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def plot_losses(train_losses, val_losses):
    '''
    Plots the training and validation losses.

    Args:
        train_losses (list): A list of training losses for each epoch.
        val_losses (list): A list of validation losses for each epoch.
    '''

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

def main():

    # Set hyperparameters
    batch_size = 8
    num_epochs = 100
    learning_rate = 1e-3

    # Get data loaders
    train_loader, val_loader = get_data_loaders(batch_size)

    # Create the model
    model = CellCounter()

    # Train the model
    trained_model, train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs, learning_rate)

    # Plot the training and validation losses
    plot_losses(train_losses, val_losses)

    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.L1Loss()
    test_loss = evaluate_model(trained_model, val_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

    # Save the model
    torch.save(trained_model.state_dict(), "cell_counter.pth")

if __name__ == "__main__":
    main()