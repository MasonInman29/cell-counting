# COM S 571X: Cell Counting Project
The code in this repository is for the final project in the course _COMS 571X Responsible AI_ at Iowa State University. The objective of this project is to automatically count the number of cells in provided cell images using machine learning.

The repository contains code for the two models tested (Cellpose-SAM and MCNN) and the code used to mitigate the limited labeled data risk and class imbalance risk. The directory also contains the provided starter model (simple CNN architecture) used for the limited labeled data risk- and class imbalance risk ablation studies.

## Installation
1. Clone the repository onto the device where you want to run the code.
2. Navigate to the cloned repository in your terminal.
3. Download the cell dataset from: https://zenodo.org/records/17088532 and place it in the cloned repository. The folder should be named "dataset".
4. Create a conda environment by running `conda env create -f environment.yml`. This will download all of the required Python dependencies.
5. Activate the Conda Environment by running `conda activate cellCounting`.

With the Conda Environment activated, you are ready to run the different Python scripts by executing `python <filename>.py`. Refer to the "File Overview" section below for an overview of the various files.

## File Overview

### Runnable Files Used to Obtain the Results

* `run_Cellpose_SAM.ipynb`: This notebook contains the code to run the pretrained CellPose-SAM model on the testing dataset.

* `train_mcnn.py`: Main runner that trains the MCNN architecture with mitigation strategies like stratified splitting, inverse frequency weighting, and utilizes the augmented dataset. Example to run is: `python train_mcnn.py --dataset-root ./final_augmented_dataset/ --batch-size 12 --epochs 100 --lr 1e-3 --seed 42 --out-root runs`. Ensure you have generated the augmented dataset first.
  
* `data_augmenter.py`: Contains the code to generate the various augmented datasets used in the data augmentation ablation studies.
* `build_final_augmented_dataset.py`: This file generates a folder with the final augmented dataset. Refer to the report for a detailed description of the final augmented dataset.
  
* `eval_by_staining.py`: Post-hoc script that runs a train MCNN model and obtains metrics needed in the report for different staining metrics. Example run: `python eval_by_staining.py --dataset-root ../dataset --model-path models/mcnn.pth --model-type mmnet --batch-size 8 --out-csv runs/log.csv`
* `eval_by_density.py`: Post-hoc script that runs a train MCNN model and obtains metrics needed in the report for different density metrics. Example run: `python eval_by_density.py --dataset-root ../dataset --model-path models/mcnn.pth --model-type mmnet --batch-size 8 --out-csv runs/log.csv`
* `main.py`: This file contains the code to run the starter code model and do the ablation experiments. The ablation experiments were run by setting the number of epochs to 20 and then uncommenting the line corresponding to what augmentation technique that should be tested. Make sure to run `data_augmenter.py` first.
* `generate_predictions.py`:

### Supporting Files 
* `model.py`: defines the architectures of the provided starter code model (simple CNN architecture) and the MCNN.   
* `dataset_handler.py`:

## The Dataset
The project utilizes the [CellFMCount dataset](https://zenodo.org/records/17088532), which comprises 3,023 fluorescence microscopy images from immunocytochemistry experiments involving neural progenitor cells. Each image is associated with a particular stain and has a corresponding CSV file containing the annotated positions of the cells of that stain in the image. The annotated positions can be viewed as the approximate center of the annotated cell. The CSV file has the following structure, where the column specifies the distance from the upper left corner in pixels:
```
X,Y
100,200
300,400
...
```
