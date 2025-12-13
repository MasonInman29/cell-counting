# COM S 571X: Cell Counting Project
The code in this repository is for the final project in the course _COMS 571X Responsible AI_ at Iowa State University. The objective of this project is to automatically count the number of cells in provided cell images using machine learning.

The repository contains code for the two models tested (Cellpose-SAM and MCNN) and the code used to mitigate the limited labeled data risk and class imbalance risk. The directory also contains the provided starter model (simple CNN architecture) used for the limited labeled data risk- and class imbalance risk ablation studies.

## Installation
1. Clone the repository onto the device where you want to run the code.
2. Navigate to the cloned repository in your terminal.
3. Download the cell dataset from: https://zenodo.org/records/17088532 and place it in the cloned repository. The folder should be namned "dataset".
4. Create a conda environment by running `conda env create -f environment.yml`. This will download all of the required python dependencies.
5. Activate the Conda Environment by running `conda activate cellCounting`.

With the Conda Environment activated, you are ready to run the differnt python scripts by executing `python <filename>.py`. See the section "File Overview" below for an overview of the different files.

## File Overview

### Runnable Files Used to Obtain the Results

* `train_mcnn.py`:  
* `data_augmenter.py`: Contains functions to generate the various augmented datasets used in the data augmentation ablation studies.
* `build_final_augmented_dataset.py`: This file generates a folder with the final augmented dataset. See the report for a description of the final augmented dataset.
* `eval_by_staining.py`:
* `main.py`: This file contains the code to run the starter code model and do the ablation experiments. The ablation experiments were run by setting the number of epochs to 20 and then uncommenting the line corresponding to what augmentation technique should be tested. Make sure to run `data_augmenter.py` first.
* `generate_predictions.py`:

### Supporting Files 
* `model.py`: defines the architectures of the provided starter code model (simple CNN architecture) and the MCNN.   
* `dataset_handler.py`:

## The Dataset
The project uses the [CellFMCount dataset](https://zenodo.org/records/17088532) with 3,023 fluorescence microscopy images from immunocytochemistry experiments involving neural progenitor cells. Each image is associated with a particular stain and has a corresponding CSV file containing the annotated positions of the cells of that stain in the image. The annotated positions can be viewed as the approximate center of the annotated cell. The CSV file has the following structure where the column specifies the distance from the upper left corner in pixels:
```
X,Y
100,200
300,400
...
```
