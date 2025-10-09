# COM S 571X: Cell Counting Project
The following repository contains example python code for automated cell counting. It is structured as follows:
- `main.py`: The main script that runs training and testing of the model.
- `model.py`: The model architecture.
- `dataset_handler.py`: The dataset handler class.

## Requirements
- Python 3.8+
- PyTorch 1.13+
- Pillow 8.1+
- Numpy 1.19+
- Pandas 1.2+

## The Dataset
The dataset used in this project is the IDCIA v2 dataset. It contains microscopic images of Adult Hippocampal Progenitor Cells (AHPCs) and their corresponding ground truth cell counts. Each image is 600x800 pixels and contains a varying number of cells. For every image, the location of each cell is provided in a CSV file. A typical ground truth CSV file is shown below:
```
X,Y
100,200
300,400
...
```
The first row contains the column names, and each subsequent row contains the x and y coordinates of a cell. To obtain the cell count, the number of rows in the CSV file is counted. The training set of the dataset containing 250 images will be provided to you. A separate test set containing 108 images will be held out for evaluation. It is important to note that the test set will not be provided to you, and you will be required to submit your predictions on the test set to be evaluated. For hyperparameter tuning, you can split the training set into a training and validation sets.

> Note that in the code provided, the images are resized to 256x256 pixels. It is up to you to decide whether to use the images in their original size or resize them to a different size.

## The Model
As a starting point, we provide you with a simple CNN model that takes an image as input and outputs the cell count. Remember that the model provided in `model.py` is a simple model and the hyperparameters are not tuned. You are encouraged to experiment with different architectures and hyperparameters to improve the performance of the model. A good starting point would be to use a pre-trained model and fine-tune it on the dataset.

Another important thing to note is that the fully connected layer in the model assumes that the input image size is 256x256 pixels. If you decide to use a different image size, you will need to adjust the size of the fully connected layer accordingly.

## Evaluation
You can use different evaluation metrics to evaluate the performance of your model. Common evaluation metrics for regression tasks include Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE). You can use any of these metrics to evaluate your model. 

## Submission
Once you have trained your model, you will need to make predictions on the test set and submit your predictions in a CSV file. The CSV file should contain two columns: `Image` and `Cell Count`. The `Image` column should contain the name of the image file, and the `Cell Count` column should contain the predicted cell count for that image. Here is an example of how the CSV file should look:
```
Image,Cell Count
image_001.tiff,10
image_002.tiff,20
...
```

## Reminders
- The dataset is provided to you for the purpose of this project only. You are not allowed to distribute the dataset or use it for any other purpose.


