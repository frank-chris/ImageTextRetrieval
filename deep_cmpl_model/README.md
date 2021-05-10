# Deep Cross-Modal Projection Matching Model

> The way to load, train and test this model has been shown in the deep_cmpl_jupyter.ipynb file. The code has also been well documented detailing the use of every function.

<br>


## Directory Structure
1. The code folder contains the source code for the model
    * The image and text models are stored in the models directory. These include mobilenet, efficientnet and bilstm.
    * The scripts folder contains the scripts for training and testing the model.
    * The utils folder contains metric calculation functions and other helper functions
    * The datasets folder contains `data.sh` which preprocesses the dataset and stores it in a pickled format. The  `fashion.py` file contains the class for creating the dataset object.
2. The data folder contains dataset related files
    * The processed_data folder contains the pickled dataset
    * Images.csv is the file storing image paths and corresponding captions
    * Reid_raw.json is a file accepted as input by process.py


<br>

## Instructions to run
1. Set the appropriate path for storing checkpoints in `trainer.py` and `tester.py` scripts. Specifically, you need to change the BASE_ROOT_2 parameter in these files. Set it to a google folder to which you have access.
2. The sample code in the jupyter notebook loads the Indian Fashion Dataset. For training the model on some other dataset, you need to change `make_json.py`, `images.csv` and `preprocess.py` appropriately.
3. Execute the instructions in the notebook.



